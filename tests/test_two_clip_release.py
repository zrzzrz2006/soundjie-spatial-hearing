"""Release integrity tests. Fixtures are retained on F:, never on the board.

Synthetic ELF containers exercise the real parsers and package builder. Only
the platform readelf invocation is substituted in portable fixture tests; a
separate test inspects the pinned base's real ARM binary with native readelf.
"""
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ARTIFACTS = Path(os.environ.get("SOUNDJIE_RELEASE_TEST_ROOT",
    "F:/MLK/DPU_ZYNQ7000/artifacts/two-clip-release-builder-20261101a" if os.name == "nt"
    else "/mnt/f/MLK/DPU_ZYNQ7000/artifacts/two-clip-release-builder-20261101a"))
CONTRACT = "soundjie-alert5bg-v3"
LABELS = ["siren", "car_horn", "doorbell", "dog_bark", "speech", "background_unknown"]
KERNEL = "soundjie_alert5bg"
THRESHOLDS = [.525, .9, .575, .6, .675]
ABI = """Class: ELF32
Data: 2's complement, little endian
Machine: ARM
Flags: 0x5000400, Version5 EABI, hard-float ABI
Tag_CPU_arch: v7
Tag_ABI_VFP_args: VFP registers
Requesting program interpreter: /lib/ld-linux-armhf.so.3
Shared library: [libc.so.6]
GLIBC_2.17
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def js(data):
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def elf(sections, kind=1):
    """Small, structurally valid ELF32 ARM fixtures, not executable programs."""
    names = b"\0.shstrtab\0" + b"".join(n.encode()+b"\0" for n in sections)
    data = bytearray(52)
    headers = [(0,)*10]
    for name, payload in [(".shstrtab", names), *sections.items()]:
        offset = len(data)
        data.extend(payload)
        headers.append((names.index(name.encode()+b"\0"), 3 if name == ".shstrtab" else 1,
                        0, 0, offset, len(payload), 0, 0, 1, 0))
    shoff = len(data)
    for header in headers:
        data.extend(struct.pack("<10I", *header))
    ident = b"\x7fELF\x01\x01\x01"+b"\0"*9
    struct.pack_into("<16sHHIIIIIHHHHHH", data, 0, ident, kind, 40, 1, 0, 0, shoff,
                     0x5000000 if kind == 1 else 0x5000400, 52, 0, 0, 40, len(headers), 1)
    return bytes(data)


def model_sections():
    strings = b"\0DPU_1152FA\0dnnc version v2.05\nDPU Target : v1.4.0\n\0conv1\0conv2\0conv3\0conv4\0fc8\0"
    metadata = [strings.index(b"DPU_"), strings.index(b"dnnc"), 0, 5, 52]+[0]*11
    tensors = [(2,64,96,1,0,6144,8,7,1,0,0,0,0)]
    tensors += [(1,1,1,1,0,1,8,4,1,0,0,0,0)]*4
    tensors += [(4,1,1,6,0,6,8,3,6,0,0,0,0)]
    nodes = []
    for name in ("conv1", "conv2", "conv3", "conv4", "fc8"):
        row = [0]*40
        row[0:2] = [1, strings.index(name.encode()+b"\0")]
        nodes.append(row)
    fc = nodes[-1]
    fc[18:22] = [0,6,8,12]
    fc[23:28] = [2,6,1,1,1]
    fc[28:32] = [6,13824,8,10]
    fc[33:40] = [2,6,96,4,6,0,0]
    contents = {"strtab": strings, "metadata": struct.pack("<16I", *metadata),
                "tensor": b"".join(struct.pack("<13I", *r) for r in tensors),
                "node": b"".join(struct.pack("<40I", *r) for r in nodes),
                "parameter": b"\1"*(6+13824), "code": b"\1"*16}
    return {".deephi."+key+"."+KERNEL: value for key,value in contents.items()}


def make_tar(path, entries, prefix="sj-alert4-release", extra=()):
    with tarfile.open(path, "w:gz") as archive:
        for name, (data, mode) in entries.items():
            item = tarfile.TarInfo(prefix+"/"+name)
            item.size = len(data)
            item.mode = mode
            archive.addfile(item, io.BytesIO(data))
        for item, data in extra:
            archive.addfile(item, io.BytesIO(data))


class TwoClipReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix="release-tests-", dir=ARTIFACTS))
        print("Retained release fixtures:", cls.work, flush=True)

    def setUp(self):
        spec = importlib.util.find_spec("scripts.host.build_soundjie_two_clip_release")
        self.assertIsNotNone(spec, "two-clip release builder is not implemented")
        self.m = importlib.import_module(spec.name)
        self.root = Path(tempfile.mkdtemp(prefix=self._testMethodName+"-", dir=self.work))
        self.repo = self.root/"repo"
        self.repo.mkdir()
        model = elf(model_sections())
        self.model = self.root/"dpu_soundjie_alert5bg.elf"
        self.model.write_bytes(model)
        self.model_sha = sha(model)
        frontend = b'{"frontend":"exact base bytes"}\r\n'
        self.programs = {}
        for kind, name in (("runtime","soundjie_d4_runtime"),("ui","soundjie_hdmi_ui"),("classifier","soundjie_d4_classifier")):
            folder = self.root/kind
            folder.mkdir()
            sections = {".rodata": b"--tv-bank\0--log-max-bytes\0soundjie-alert5bg-v3\0"+self.model_sha.encode()+b"\0dpu_soundjie_alert5bg.elf\0",
                        ".symtab": b"\0"*16}
            if kind == "classifier":
                sections.update(model_sections())
            (folder/name).write_bytes(elf(sections, 2))
            report = dict(MACHINE="ARM",CPU_ARCH="v7",HARD_FLOAT="yes",
                          ELF_INTERPRETER="/lib/ld-linux-armhf.so.3",MAX_GLIBC="2.17",DT_NEEDED="libc.so.6",
                          SHA256=sha((folder/name).read_bytes()))
            if kind == "classifier":
                report.update(MODEL_SHA256=self.model_sha,FRONTEND_SHA256=sha(frontend),PHYSICAL_OUTPUTS="6",STORAGE_OUTPUTS="6",
                              KERNEL_NAME=KERNEL,MODEL_BASENAME=self.model.name)
            (folder/"build-report.txt").write_text("".join(k+"="+v+"\n" for k,v in report.items()),encoding="utf-8")
            self.programs[kind] = folder
        classes = dict(contract=CONTRACT,labels=LABELS,physical_output_count=6,logical_output_count=6,background_index=5,speech_index=4)
        for name, data in {"config/soundjie_alert5bg_classes.json": js(classes),
                           "assets/hdmi/soundjie_two_clip_glyph_atlas.bin": b"SJGA1\0\0\0"+struct.pack("<II",1,1)+struct.pack("<IB3x16H",ord("A"),8,*([0]*16)),
                           "scripts/board/run_soundjie_d4_session.sh": b"#!/bin/sh\r\n# --tv-bank soundjie-alert5bg-v3 --log-max-bytes\r\nexit 0\r\n",
                           "scripts/board/run_soundjie_tv_session.sh": b"#!/bin/sh\nexport SOUNDJIE_TV_ENABLE=1\nexec sh run_soundjie_d4_session.sh\n"}.items():
            p = self.repo/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(data)
        self.bank = self.root/"bank"
        self.bank.mkdir()
        features = b"SJTVF2\0\0"+struct.pack("<II",400,32)+struct.pack("<12800f",*([1.]*12800))
        data = {"bank.tsv":b"a\tambulance_traffic\tambulance/reference.features\tambulance/cues.tsv\nb\tdialogue\tdialogue/reference.features\tdialogue/cues.tsv\n",
                "ambulance/reference.features":features,"dialogue/reference.features":features,
                "ambulance/cues.tsv":b"a\t0\t1.8\tA\tAmbulance\n","dialogue/cues.tsv":b"b\t0\t1.8\tC\tDialogue\n",
                "traffic.bin":b"SJTF1\0\0\0"+struct.pack("<III",96,48,3)+struct.pack("<4995f",*([1.]*4995)),
                "dialogue-horn-start.txt":b".4\n","dialogue-horn-end.txt":b"1.4\n"}
        for name,payload in data.items():
            path=self.bank/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(payload)
        self.original = {"runtime/"+name:(b"old-"+name.encode(),0o755) for name in
                         ("soundjie_d4_runtime","soundjie_d4_classifier","soundjie_hdmi_ui","xvf3800_doa","dpu-4.14.0-xilinx.ko")}
        self.original.update({"audio/bin/arecord":(b"unchanged capture",0o755),"audio/lib/libasound.so.2":(b"unchanged ALSA",0o644),
                              "dnndk/pkgs/lib/libn2cube.so":(b"unchanged DNNDK",0o644),"dnndk/pkgs/lib/libhineon.so":(b"unchanged NEON",0o644),
                              "licenses/license.txt":(b"unchanged license",0o644),"config/logmel_frontend_v1.json":(frontend,0o644),
                              "config/soundjie_d4_runtime.tsv":(b"exact DoA/display\r\n",0o644),
                              "model/dpu_soundjie_d3.elf":(b"old model",0o644),"model/storage-layout.json":(b"old padding",0o644),
                              "model/thresholds-fixed.json":(b"old receipt",0o644),"manifest/release.json":(b"{}",0o644),
                              "manifest/model-package.json":(b"old manifest",0o644),"manifest/ui-update.json":(b"old UI manifest",0o644),
                              "probe/reference.json":(b"old expected logits",0o644),"probe/window-00.s16":(b"old probe",0o644),
                              "evidence/regression.log":(b"old tests",0o644),"tv/reference.features":(features,0o644),
                              "tv/cues.tsv":(b"old cues",0o644),"tv/traffic.bin":(data["traffic.bin"],0o644),
                              "tv/clip_config.json":(b"old TV metadata",0o644),"tv/label_policy.json":(b"old scene",0o644),
                              "board/probe_soundjie_alert4.sh":(b"old model probe",0o755),
                              "board/run_soundjie_d4_session.sh":(b"old launcher",0o755),"board/run_soundjie_tv_session.sh":(b"old launcher",0o755),
                              "assets/hdmi/soundjie_glyph_atlas.bin":(b"old atlas",0o644)})
        for name,value in (("class-contract.txt",b"old"),("alert-thresholds.txt",b"old"),("classes.json",b"old"),("hop-samples.txt",b"6400\n"),("log-policy.txt",b"bounded-jsonl-v1\n")):
            self.original["config/"+name]=(value,0o644)
        self.original["SHA256SUMS"]=("".join(sha(data)+"  "+name+"\n" for name,(data,_) in sorted(self.original.items())).encode(),0o644)
        self.base=self.root/"base.tgz"
        make_tar(self.base,self.original)
        self.base_sha=sha(self.base.read_bytes())
        self.pin_bank()
        self.receipt=self.root/"receipt.json"
        self.receipt.write_bytes(js(dict(contract=CONTRACT,elf_sha256=self.model_sha,kernel=KERNEL,
            physical_output_count=6,logical_output_count=6,labels=LABELS,background_index=5,
            alert_thresholds=THRESHOLDS,foreground_thresholds=THRESHOLDS,checkpoint_sha256="1"*64,
            board_tested=False,independent_acceptance=False)))
        self.out=self.root/"out"

    def pin_bank(self):
        pins={p.relative_to(self.bank).as_posix():sha(p.read_bytes()) for p in self.bank.rglob("*") if p.is_file() and p.name!="bank-manifest.json"}
        (self.bank/"bank-manifest.json").write_bytes(js(dict(schema=1,files=pins,old_archive_sha256=self.base_sha)))

    def build(self,**overrides):
        args=dict(base=self.base,runtime_dir=self.programs["runtime"],ui_dir=self.programs["ui"],classifier_dir=self.programs["classifier"],
                  model_elf=self.model,model_sha256=self.model_sha,thresholds=THRESHOLDS,bank_assets_dir=self.bank,output_dir=self.out,
                  training_receipt=self.receipt,repo_root=self.repo)
        args.update(overrides)
        with patch.object(self.m,"BASE_SHA",self.base_sha),patch.object(self.m,"readelf",return_value=ABI):
            return self.m.build_release(**args)

    def test_complete_archive_preserves_dependencies_modes_and_inventory(self):
        original_sources={p:sha(p.read_bytes()) for p in self.root.rglob("*") if p.is_file()}
        result=self.build()
        self.assertFalse(result["board_tested"])
        archive=self.m.read_archive(self.out/"SJ6V1.TGZ",result["archive_sha256"],"sj-two-clip-release")
        for name in ("audio/bin/arecord","audio/lib/libasound.so.2","runtime/xvf3800_doa","runtime/dpu-4.14.0-xilinx.ko",
                     "dnndk/pkgs/lib/libn2cube.so","config/logmel_frontend_v1.json","config/soundjie_d4_runtime.tsv","licenses/license.txt"):
            self.assertEqual(tuple(archive[name]),self.original[name])
        for removed in ("model/dpu_soundjie_d3.elf","tv/reference.features","tv/cues.tsv","tv/clip_config.json","tv/label_policy.json",
                        "manifest/ui-update.json","manifest/model-package.json","probe/reference.json","probe/window-00.s16","board/probe_soundjie_alert4.sh"):
            self.assertNotIn(removed,archive)
        self.assertEqual(archive["model/dpu_soundjie_alert5bg.elf"].data,self.model.read_bytes())
        self.assertEqual(archive["config/class-contract.txt"].data,(CONTRACT+"\n").encode())
        self.assertEqual(archive["config/alert-thresholds.txt"].data,b"0.525,0.9,0.575,0.6,0.675\n")
        self.assertEqual(archive["config/hop-samples.txt"].data,b"6400\n")
        self.assertEqual(archive["config/log-policy.txt"].data,b"bounded-jsonl-v1\n")
        self.assertEqual(archive["tv/bank.tsv"].data,(self.bank/"bank.tsv").read_bytes())
        self.assertEqual(archive["runtime/soundjie_d4_classifier"].mode,0o755)
        self.assertNotIn(b"\r",archive["board/run_soundjie_d4_session.sh"].data)
        self.m.verify_checksums(archive)
        layout=json.loads(archive["model/storage-layout.json"].data)
        self.assertEqual(layout["physical_output_count"],6)
        self.assertEqual(layout["storage_output_count"],6)
        self.assertIsNone(layout["padding"])
        manifest=json.loads(archive["manifest/release.json"].data)
        self.assertEqual(set(manifest["payload_sha256"]),set(archive)-{"SHA256SUMS","manifest/release.json"})
        self.assertEqual(manifest["kernel"],KERNEL)
        self.assertEqual(manifest["model_sha256"],self.model_sha)
        self.assertEqual((self.out/"SJ6V1.SHA").read_text(),result["archive_sha256"]+"  SJ6V1.TGZ\n")
        for source,digest in original_sources.items(): self.assertEqual(sha(source.read_bytes()),digest)

    def test_measured_hop_is_bound_to_package_and_manifest(self):
        for invalid in (7999,True,0):
            with self.assertRaises(ValueError):
                self.build(hop_samples=invalid)
            self.assertFalse(self.out.exists())
        result=self.build(hop_samples=8000)
        archive=self.m.read_archive(self.out/"SJ6V1.TGZ",result["archive_sha256"],"sj-two-clip-release")
        self.assertEqual(archive["config/hop-samples.txt"].data,b"8000\n")
        self.assertEqual(json.loads(archive["manifest/release.json"].data)["hop_samples"],8000)
        self.assertEqual(result["hop_samples"],8000)

    def test_create_only_and_invalid_inputs_leave_no_destination(self):
        for kwargs in (dict(model_sha256="0"*64),dict(thresholds=[.5]*4),dict(thresholds=[.5]*4+[float("nan")]),dict(thresholds=[1.1]*5)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): self.build(**kwargs)
            self.assertFalse(self.out.exists())
        self.out.mkdir();(self.out/"mine").write_bytes(b"keep")
        with self.assertRaises(FileExistsError): self.build()
        self.assertEqual((self.out/"mine").read_bytes(),b"keep")

    def test_exact_report_values_model_kernel_and_hash_are_required(self):
        report=self.programs["classifier"]/"build-report.txt"
        original=report.read_text()
        for field,bad in (("PHYSICAL_OUTPUTS=6","PHYSICAL_OUTPUTS=5"),("STORAGE_OUTPUTS=6","STORAGE_OUTPUTS=8"),
                          ("KERNEL_NAME="+KERNEL,"KERNEL_NAME=soundjie_d3"),("MODEL_BASENAME="+self.model.name,"MODEL_BASENAME=dpu_soundjie_d3.elf"),
                          ("MODEL_SHA256="+self.model_sha,"MODEL_SHA256="+"0"*64),("SHA256=","SHA256=00")):
            report.write_text(original.replace(field,bad))
            with self.subTest(field=field),self.assertRaises(ValueError): self.build()
            self.assertFalse(self.out.exists())
        report.write_text(original+"PHYSICAL_OUTPUTS=6\n")
        with self.assertRaises(ValueError): self.build()

    def test_mismatched_linked_model_sections_are_rejected_despite_rehashed_report(self):
        binary=self.programs["classifier"]/"soundjie_d4_classifier"
        before=sha(binary.read_bytes())
        changed=bytearray(binary.read_bytes());i=changed.index(b"\1"*100);changed[i]=2
        binary.write_bytes(changed)
        report=binary.parent/"build-report.txt";report.write_text(report.read_text().replace(before,sha(changed)))
        with self.assertRaises(ValueError): self.build()
        self.assertFalse(self.out.exists())

    def test_abi_and_structural_elf_checks_reject_wrong_targets(self):
        self.assertEqual(self.m.audit_abi(ABI)["max_glibc"],"2.17")
        for text in (ABI.replace("Machine: ARM","Machine: X86-64"),ABI.replace("v7","v8"),ABI+"GLIBC_2.34\n",
                     ABI.replace("VFP registers","base PCS"),ABI+"Shared library: [libstdc++.so.6]\n"):
            with self.assertRaises(ValueError): self.m.audit_abi(text)
        for raw in (b"not ELF",elf(model_sections())[:52],elf(model_sections()).replace(b"\x28\x00",b"\x3e\x00",1)):
            with self.assertRaises(ValueError): self.m.elf_sections(raw,relocatable=True)

    def test_bank_pins_and_semantics_and_receipt_bindings(self):
        cue=self.bank/"dialogue/cues.tsv";original=cue.read_bytes()
        cue.write_bytes(b"changed")
        with self.assertRaises(ValueError): self.build()
        self.pin_bank()
        with self.assertRaises(ValueError): self.build()
        cue.write_bytes(original);self.pin_bank()
        horn=self.bank/"dialogue-horn-end.txt";horn.write_bytes(b"999\n");self.pin_bank()
        with self.assertRaises(ValueError): self.build()
        horn.write_bytes(b"1.4\n");self.pin_bank()
        receipt=json.loads(self.receipt.read_bytes());receipt["elf_sha256"]="f"*64;self.receipt.write_bytes(js(receipt))
        with self.assertRaises(ValueError): self.build()
        self.assertFalse(self.out.exists())

    def test_archive_paths_duplicates_links_modes_and_checksums_are_validated(self):
        for name,kind,mode in (("../escape",tarfile.REGTYPE,0o644),("/absolute",tarfile.REGTYPE,0o644),
                               ("sj-alert4-release/a/../b",tarfile.REGTYPE,0o644),("sj-alert4-release/a",tarfile.SYMTYPE,0o644),
                               ("sj-alert4-release/a",tarfile.LNKTYPE,0o644),("sj-alert4-release/a",tarfile.FIFOTYPE,0o644),
                               ("sj-alert4-release/a",tarfile.REGTYPE,0o4755),("sj-alert4-release/a",tarfile.REGTYPE,0o666),
                               ("sj-alert4-release/runtime/xvf3800_doa",tarfile.REGTYPE,0o644)):
            item=tarfile.TarInfo(name);item.type=kind;item.mode=mode;item.linkname="/outside" if kind in (tarfile.SYMTYPE,tarfile.LNKTYPE) else ""
            path=self.root/"malicious.tgz";make_tar(path,self.original,extra=[(item,b"")])
            with self.subTest(name=name,kind=kind,mode=mode),self.assertRaises(ValueError):
                self.m.read_archive(path,sha(path.read_bytes()),"sj-alert4-release")
        bad=dict(self.original);bad["audio/bin/arecord"]=(b"corrupt",0o755)
        make_tar(self.base,bad)
        with self.assertRaises(ValueError): self.m.read_archive(self.base,sha(self.base.read_bytes()),"sj-alert4-release")

    def test_failed_archive_readback_never_writes_completion_marker(self):
        with patch.object(self.m,"write_archive",side_effect=lambda path,payload: Path(path).write_bytes(b"broken archive")):
            with self.assertRaises((ValueError,tarfile.TarError,OSError,EOFError)): self.build()
        self.assertTrue(self.out.exists())
        self.assertFalse((self.out/"SJ6V1.SHA").exists())

    def test_source_links_are_refused_without_os_changes(self):
        link=self.root/"model-link.elf"
        try: link.symlink_to(self.model)
        except (OSError,NotImplementedError): self.skipTest("filesystem does not permit links; OS settings unchanged")
        with self.assertRaises(ValueError): self.build(model_elf=link)
        self.assertFalse(self.out.exists())

    def test_real_pinned_base_and_readelf_inspection(self):
        if not self.m.DEFAULT_BASE.exists(): self.skipTest("real base not mounted")
        entries=self.m.read_archive(self.m.DEFAULT_BASE,self.m.BASE_SHA,"sj-alert4-release")
        raw=entries["runtime/soundjie_d4_runtime"].data
        self.m.elf_sections(raw,relocatable=False)
        path=self.root/"actual-arm-runtime";path.write_bytes(raw)
        report=self.m.audit_abi(self.m.readelf(path))
        self.assertEqual(report["max_glibc"],"2.17")


if __name__ == "__main__":
    unittest.main()
