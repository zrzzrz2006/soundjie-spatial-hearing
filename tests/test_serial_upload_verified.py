"""In-memory protocol tests: no physical serial port, board or boot interaction."""
import ast
import base64
import hashlib
import importlib
import io
import unittest
import zlib


def receiver_module(module):
    namespace = {"__name__": "protocol_fixture"}
    exec(compile(module.RECEIVER_SOURCE, "<receiver>", "exec"), namespace)
    return namespace


class MemorySerial:
    """Byte-stream adapter using the production receiver core, with wire faults."""
    def __init__(self, module, data, *, lose_ack=False, corrupt=False, lose_done=False):
        self.module = module
        self.sid = "0123456789abcdef"
        self.data = data
        self.sink = io.BytesIO()
        self.core = receiver_module(module)["Core"](self.sink, len(data), hashlib.sha256(data).hexdigest())
        self.input = bytearray()
        self.output = bytearray()
        self.lose_ack, self.corrupt, self.lose_done = lose_ack, corrupt, lose_done
        self.sent = []
        self.commit_count = 0

    def write(self, raw):
        self.input.extend(raw)
        while b"\n" in self.input:
            line, _, remaining = self.input.partition(b"\n")
            self.input = bytearray(remaining)
            if not line:
                continue
            fields = line.decode("ascii").split()
            self.sent.append(fields)
            if fields[:2] != ["#S", self.sid]:
                continue
            command = fields[2:]
            if command == ["Q"] or command == ["X"]:
                self.output.extend(("SV1 "+self.sid+" RESTORED 0\n").encode())
                continue
            if command[0] == "D" and self.corrupt:
                self.corrupt = False
                command[-1] = base64.b64encode(b"bad").decode()
            previous = self.core.done
            response = self.core.feed(command)
            if self.core.done and not previous:
                self.commit_count += 1
            if response.startswith("A ") and self.lose_ack:
                self.lose_ack = False
                continue
            if response.startswith("DONE ") and self.lose_done:
                self.lose_done = False
                continue
            # Kernel chatter is unrelated to acknowledgements and is retained.
            self.output.extend(b"[ 14.0] uart: input overrun\r\n")
            self.output.extend(("SV1 "+self.sid+" "+response+"\r\n").encode())
        return len(raw)

    def read(self, count=1):
        data = bytes(self.output[:count])
        del self.output[:count]
        return data


class SerialUploadTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec("scripts.host.serial_upload_verified")
        self.assertIsNotNone(spec, "verified stop-and-wait uploader is not implemented")
        self.m = importlib.import_module(spec.name)
        self.sid = "0123456789abcdef"

    def data_fields(self, sequence, data):
        return ["D", str(sequence), "%08x" % (zlib.crc32(data)&0xffffffff), base64.b64encode(data).decode()]

    def test_lost_ack_and_corrupt_frame_retry_without_duplicate_append(self):
        data = bytes(range(256))*7+b"tail"
        board = MemorySerial(self.m,data,lose_ack=True,corrupt=True,lose_done=True)
        log = io.BytesIO()
        transport = self.m.Link(board,log,self.sid,timeout=.005,retries=5,tx_gap=0)
        result = self.m.send_file(transport,io.BytesIO(data),len(data),hashlib.sha256(data).hexdigest())
        self.assertEqual(board.sink.getvalue(),data)
        self.assertEqual(board.commit_count,1)
        self.assertEqual(result["sha256"],hashlib.sha256(data).hexdigest())
        self.assertEqual(result["size"],len(data))
        self.assertGreater(sum(row[2:4] == ["D","0"] for row in board.sent),1)
        self.assertIn(b"input overrun",log.getvalue())
        self.assertNotIn(base64.b64encode(data[:512]),log.getvalue())

    def test_receiver_rejects_mutated_duplicate_out_of_order_and_oversize(self):
        data = b"a"*512+b"b"*5
        core = receiver_module(self.m)["Core"](io.BytesIO(),len(data),hashlib.sha256(data).hexdigest())
        self.assertTrue(core.feed(self.data_fields(1,b"b"*5)).startswith("N "))
        self.assertTrue(core.feed(self.data_fields(0,b"a"*513)).startswith("N "))
        ack = core.feed(self.data_fields(0,b"a"*512))
        self.assertTrue(ack.startswith("A 0 "))
        self.assertEqual(core.feed(self.data_fields(0,b"a"*512)),ack)
        self.assertTrue(core.feed(self.data_fields(0,b"z"*512)).startswith("N "))
        self.assertEqual(core.sink.getvalue(),b"a"*512)

    def test_final_receipt_requires_full_size_and_readback_sha(self):
        namespace = receiver_module(self.m)
        core = namespace["Core"](io.BytesIO(),3,hashlib.sha256(b"abc").hexdigest())
        self.assertTrue(core.feed(["F"]).startswith("N "))
        core.feed(self.data_fields(0,b"abc"))
        core.sink.seek(0);core.sink.write(b"xyz")
        with self.assertRaises(ValueError): core.feed(["F"])
        self.assertFalse(core.done)
        empty = namespace["Core"](io.BytesIO(),0,hashlib.sha256(b"").hexdigest())
        self.assertTrue(empty.feed(["F"]).startswith("DONE 0 "))

    def test_receiver_and_host_path_guards_agree(self):
        guard = receiver_module(self.m)["valid_path"]
        for name in ("/run/sj6-payload.tgz","/run/sj6-transfer/payload.tgz","/run/sj6-transfer/sub/file.bin"):
            self.assertTrue(guard(name));self.assertEqual(self.m.validate_target(name),name)
        for name in ("/mnt/sd/payload","/run/file","/run/sj6-transfer","/run/sj6-x/file","/run/sj6-transfer/../sd",
                     "/run/sj6-transfer/a//b","/run/sj6-transfer/./b","/run/sj6-transfer/a;reboot",
                     "/run/sj6-transfer/a\nreboot","/run/sj6-transfer/a'", "relative", "/run/sj6-"):
            with self.subTest(name=name):
                self.assertFalse(guard(name))
                with self.assertRaises(ValueError): self.m.validate_target(name)

    def test_bootstrap_line_limits_and_python35_syntax(self):
        ast.parse(self.m.RECEIVER_SOURCE,feature_version=(3,5))
        ast.parse(self.m.BOOTSTRAP_SOURCE,feature_version=(3,5))
        self.assertLessEqual(len(self.m.RECEIVER_SOURCE.encode()),4096)
        self.assertLessEqual(len(self.m.bootstrap_blob()),4096)
        for line in self.m.bootstrap_frames(self.sid):
            self.assertLessEqual(len(line.rstrip(b"\n")),76)
        command = self.m.bootstrap_command(self.sid,"/run/sj6-transfer/payload.tgz",100,"a"*64)
        self.assertLess(len(command),4096)
        self.assertNotIn(b"\x03",command)
        self.assertNotIn(b"\x04",command)
        self.assertIn(b"stty",command)

    def test_bootstrap_lost_ack_is_idempotent_and_sha_is_checked(self):
        space = {"__name__":"fixture"}
        exec(compile(self.m.BOOTSTRAP_SOURCE,"<bootstrap>","exec"),space)
        blob = self.m.bootstrap_blob()
        boot = space["Boot"](hashlib.sha256(blob).hexdigest())
        frames = list(self.m.bootstrap_frames(self.sid))
        first = frames[0].decode().split()[2:]
        self.assertEqual(boot.feed(first),boot.feed(first))
        for line in frames[1:]: boot.feed(line.decode().split()[2:])
        self.assertEqual(boot.feed(["E"]),blob)
        bad = space["Boot"]("0"*64)
        for line in frames: bad.feed(line.decode().split()[2:])
        with self.assertRaises(ValueError): bad.feed(["E"])

    def test_timeouts_and_wrong_session_never_report_success(self):
        data = b"payload"
        board = MemorySerial(self.m,data)
        board.write = lambda value: len(value)
        board.output.extend(b"SV1 wrong DONE 7 "+hashlib.sha256(data).hexdigest().encode()+b"\n")
        link = self.m.Link(board,io.BytesIO(),self.sid,timeout=.002,retries=2,tx_gap=0)
        with self.assertRaises(TimeoutError): self.m.send_file(link,io.BytesIO(data),len(data),hashlib.sha256(data).hexdigest())

    def test_host_rejects_changed_source_and_forged_final_sha(self):
        board = MemorySerial(self.m,b"abc")
        link = self.m.Link(board,io.BytesIO(),self.sid,timeout=.002,retries=2,tx_gap=0)
        with self.assertRaises(ValueError): self.m.send_file(link,io.BytesIO(b"abc"),3,"0"*64)
        self.assertFalse(any(row[2:] == ["F"] for row in board.sent))


if __name__ == "__main__":
    unittest.main()
