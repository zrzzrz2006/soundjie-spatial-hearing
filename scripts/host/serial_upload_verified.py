#!/usr/bin/env python3
"""Verified stop-and-wait RAM uploader; never interrupts boot or writes SD.

Run only after closing the old console and confirming an idle Linux shell:
  python -B -m scripts.host.serial_upload_verified --port COM13 --source FILE \
    --sha256 EXACT_SOURCE_SHA256 --target /run/sj6-transfer/payload.tgz \
    --log NEW_TRANSCRIPT.log --linux-shell-confirmed

Defaults: 115200 baud, 512 decoded bytes/frame, 128 TX bytes then 15 ms delay,
2 s ACK timeout, 8 attempts. --tx-delay 0.004 may be selected by the operator
after a small verified transfer. There is NO per-slice flush. Bootstrap frames
are <=76 columns and also stop-and-wait. Receiver source and compressed bootstrap
are <=4 KiB; the compressed AND expanded SHA256 are checked before execution.

The only executed board code is the fixed, hash-verified receiver below. Frame
lines are shell comments if the receiver has already exited. No Ctrl-C, break,
autoboot interruption, automatic port retry or legacy-console modification.
Partial files are private staging files; final publication is an exclusive
hard link after size/SHA readback. Existing targets are never overwritten.
The shell restores its saved stty state on receiver completion/error/idle timeout.
Host failures attempt a protocol abort, except after observing a boot banner.
Success requires the board's exact DONE receipt and RESTORED 0 acknowledgement.
RX transcript and command/frame metadata are logged; TX payload base64 is not.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import sys
import time
import zlib


# This is the actual Python 3.5 program sent to the board. Keep it small enough
# to inspect and hash before executing; tests execute its real protocol core.
RECEIVER_SOURCE = r'''import os,sys,re,hashlib,base64,zlib,signal
def valid_path(p):
 a=p.split('/')
 return bool(len(p)<=180 and re.match(r'^/run/[A-Za-z0-9_./-]+$',p) and all(x not in ('','.','..') for x in a[1:]) and ((len(a)==3 and a[2].startswith('sj6-') and len(a[2])>4 and a[2]!='sj6-transfer') or (len(a)>3 and a[2]=='sj6-transfer')))
class Core:
 def __init__(s,f,n,h):
  s.sink=f;s.size=n;s.hash=h;s.seq=0;s.count=0;s.last=None;s.done=False
 def feed(s,p):
  if p==['F']:
   if not s.done:
    if s.count!=s.size:return 'N '+str(s.seq)
    s.sink.flush();s.sink.seek(0);h=hashlib.sha256();n=0
    while True:
     b=s.sink.read(65536)
     if not b:break
     h.update(b);n+=len(b)
    if n!=s.size or h.hexdigest()!=s.hash:raise ValueError('final_hash')
    s.done=True
   return 'DONE %d %s'%(s.size,s.hash)
  try:
   if len(p)!=4 or p[0]!='D' or len(p[3])>684:raise ValueError()
   q=int(p[1]);c=p[2];b=base64.b64decode(p[3].encode('ascii'),validate=True)
   if not 0<len(b)<=512 or c!='%08x'%(zlib.crc32(b)&0xffffffff):raise ValueError()
  except Exception:return 'N '+str(s.seq)
  if (q,b)==s.last:return 'A %d %s'%(q,c)
  if s.done or q!=s.seq or len(b)!=min(512,s.size-s.count):return 'N '+str(s.seq)
  if s.sink.write(b)!=len(b):raise IOError('short_write')
  s.count+=len(b);s.seq+=1;s.last=(q,b)
  return 'A %d %s'%(q,c)
def run():
 sid,target,n,h=sys.argv[1:];f=None;part=None
 def say(t):sys.stdout.write('SV1 '+sid+' '+t+'\n');sys.stdout.flush()
 def expired(*a):raise IOError('idle_timeout')
 try:
  n=int(n)
  if not valid_path(target) or not 0<=n<=67108864 or not re.match(r'^[0-9a-f]{64}$',h):raise ValueError('arguments')
  if os.path.realpath('/run')!='/run' or not any(x.split()[1:3]==['/run','tmpfs'] for x in open('/proc/mounts')):raise ValueError('not_tmpfs')
  dev=os.stat('/run').st_dev;p='/run'
  for x in target.split('/')[2:-1]:
   p+='/'+x
   if not os.path.lexists(p):os.mkdir(p,448)
   if os.path.islink(p) or not os.path.isdir(p) or os.stat(p).st_dev!=dev:raise ValueError('parent')
  if os.path.lexists(target):raise ValueError('exists')
  candidate=target+'.part-'+sid
  fd=os.open(candidate,os.O_CREAT|os.O_EXCL|os.O_RDWR,384);part=candidate
  f=os.fdopen(fd,'w+b');core=Core(f,n,h);published=False
  signal.signal(signal.SIGALRM,expired);say('READY')
  while True:
   signal.alarm(120);line=sys.stdin.readline(1024)
   if not line:raise IOError('eof')
   p=line.split()
   if p[:2]==['#B',sid]:
    if p[2:]==['X']:break
    if p[2:]==['E']:say('READY')
    continue
   if p[:2]!=['#S',sid]:continue
   if p[2:]==['X'] or (p[2:]==['Q'] and published):break
   r=core.feed(p[2:])
   if core.done and not published:
    f.flush();os.fsync(f.fileno());os.link(part,target);published=True
   say(r)
 except Exception:
  say('ERROR receiver');return 2
 finally:
  signal.alarm(0)
  if f:f.close()
  if part:os.unlink(part)
 return 0
if __name__=='__main__':sys.exit(run())
'''

BOOTSTRAP_SOURCE = r'''import sys,hashlib,base64,zlib,signal
class Boot:
 def __init__(s,h):s.h=h;s.seq=0;s.parts=[];s.last=None
 def feed(s,p):
  if p==['E']:
   b=b''.join(s.parts)
   if hashlib.sha256(b).hexdigest()!=s.h:raise ValueError('bootstrap_hash')
   return b
  try:
   if len(p)!=3:raise ValueError()
   q=int(p[0]);c=p[1];b=base64.b64decode(p[2].encode('ascii'),validate=True)
   if not 0<len(b)<=24 or c!='%08x'%(zlib.crc32(b)&0xffffffff):raise ValueError()
  except Exception:return 'N '+str(s.seq)
  if (q,b)==s.last:return 'B %d %s'%(q,c)
  if q!=s.seq or sum(map(len,s.parts))+len(b)>4096:return 'N '+str(s.seq)
  s.parts.append(b);s.last=(q,b);s.seq+=1
  return 'B %d %s'%(q,c)
def run():
 sid,bh,rh,target,n,h=sys.argv[1:];boot=Boot(bh)
 def say(s):sys.stdout.write('SV1 '+sid+' '+s+'\n');sys.stdout.flush()
 def expired(*a):raise IOError('idle')
 signal.signal(signal.SIGALRM,expired);say('BOOT')
 try:
  while True:
   signal.alarm(120);p=sys.stdin.readline(256).split()
   if p[:2]!=['#B',sid]:continue
   if p[2:]==['X']:return
   r=boot.feed(p[2:])
   if isinstance(r,bytes):break
   say(r)
  code=zlib.decompress(r)
  if len(code)>4096 or hashlib.sha256(code).hexdigest()!=rh:raise ValueError('source_hash')
 except Exception:say('ERROR bootstrap');sys.exit(2)
 signal.alarm(0);sys.argv=['receiver',sid,target,n,h]
 exec(compile(code,'<verified-receiver>','exec'),{'__name__':'__main__'})
if __name__=='__main__':run()
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def crc(data):
    return "%08x" % (zlib.crc32(data)&0xffffffff)


def validate_target(path):
    parts = path.split("/")
    direct = len(parts) == 3 and parts[2].startswith("sj6-") and len(parts[2]) > 4 and parts[2] != "sj6-transfer"
    nested = len(parts) > 3 and parts[2] == "sj6-transfer"
    if not (len(path) <= 180 and re.fullmatch(r"/run/[A-Za-z0-9_./-]+",path) and
            all(part not in ("", ".", "..") for part in parts[1:]) and (direct or nested)):
        raise ValueError("target must be a new canonical /run/sj6-* file or /run/sj6-transfer/... file")
    return path


def bootstrap_blob():
    raw = RECEIVER_SOURCE.encode("ascii")
    blob = zlib.compress(raw,9)
    if len(raw) > 4096 or len(blob) > 4096:
        raise ValueError("receiver exceeds bootstrap bound")
    return blob


def bootstrap_frames(sid):
    blob = bootstrap_blob()
    for sequence, offset in enumerate(range(0,len(blob),24)):
        data = blob[offset:offset+24]
        yield ("#B %s %d %s %s\n" % (sid,sequence,crc(data),base64.b64encode(data).decode())).encode("ascii")


def bootstrap_command(sid, target, size, expected_sha):
    validate_target(target)
    if not re.fullmatch(r"[0-9a-f]{16}",sid) or not re.fullmatch(r"[0-9a-f]{64}",expected_sha) or not 0 <= size <= 67108864:
        raise ValueError("invalid bootstrap arguments")
    args = ["python3","-u","-c",BOOTSTRAP_SOURCE,sid,digest(bootstrap_blob()),digest(RECEIVER_SOURCE.encode()),target,str(size),expected_sha]
    # The wrapper is one shell command, not a pasted multiline shell program.
    # Escape literal newlines inside the Python argument using a verified ASCII
    # representation, keeping each canonical TTY input line below 4096 bytes.
    args[3] = "exec("+repr(BOOTSTRAP_SOURCE)+")"
    script = ('old=$(stty -g) || exit 1; '
              'finish() { rc=$?; trap - EXIT HUP INT TERM; stty "$old" || rc=125; '
              'printf "\\nSV1 '+sid+' RESTORED %s\\n" "$rc"; }; '
              'trap finish EXIT; trap "exit 130" HUP INT TERM; '
              'stty -echo -echonl icanon icrnl || exit 1; '+" ".join(shlex.quote(arg) for arg in args))
    command = ("sh -c "+shlex.quote(script)+"\n").encode("ascii")
    if len(command) >= 4096:
        raise ValueError("initial command exceeds canonical TTY line limit")
    return command


class BootDetected(RuntimeError):
    pass


class Link:
    def __init__(self, port, log, sid, *, timeout=2., retries=8, tx_gap=.015, tx_slice=128):
        self.port, self.log, self.sid = port, log, sid
        self.timeout, self.retries, self.tx_gap, self.tx_slice = timeout, retries, tx_gap, tx_slice
        self.buffer = bytearray()
        self.messages = []
        self.boot_seen = False
        self.stage = "bootstrap"

    def event(self, kind, **fields):
        self.log.write((json.dumps(dict(event=kind,**fields),sort_keys=True)+"\n").encode())
        self.log.flush()

    def _ingest(self, data):
        self.buffer.extend(data)
        if re.search(rb"U-Boot|Hit any key to stop autoboot|Zynq>",bytes(self.buffer),re.I):
            self.boot_seen = True
            self.event("BOOT_DETECTED",bytes=len(self.buffer))
            raise BootDetected("boot banner observed; no further bytes will be sent")
        while b"\n" in self.buffer:
            raw, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            if b"#S "+self.sid.encode()+b" D " in raw or re.search(rb"[A-Za-z0-9+/]{100,}={0,2}",raw):
                self.event("RX_REDACTED",bytes=len(raw),sha256=digest(raw))
            else:
                self.log.write(bytes(raw)+b"\n");self.log.flush()
            text = raw.decode("ascii","replace").strip("\r")
            prefix = "SV1 "+self.sid+" "
            if text.startswith(prefix):
                self.messages.append(text[len(prefix):])
        if len(self.buffer) > 8192:
            self.event("RX_LONG_LINE",bytes=len(self.buffer),sha256=digest(self.buffer))
            self.buffer.clear()

    def poll(self):
        self._ingest(self.port.read(1024))

    def send(self, line, *, command=False):
        if self.boot_seen:
            raise BootDetected("refusing TX after boot banner")
        if command:
            self.event("TX_COMMAND",bytes=len(line),sha256=digest(line))
            raw = line
        else:
            raw = b"\n"+line.rstrip(b"\n")+b"\n"
        for offset in range(0,len(raw),self.tx_slice):
            if getattr(self.port,"in_waiting",0):
                self.poll()
            block = raw[offset:offset+self.tx_slice]
            if self.port.write(block) != len(block):
                raise IOError("short serial write")
            if self.tx_gap and offset+self.tx_slice < len(raw):
                time.sleep(self.tx_gap)
        # No flush per slice: ACK reception is the pacing/drain boundary.

    def wait(self, predicate):
        until = time.monotonic()+self.timeout
        while True:
            while self.messages:
                message = self.messages.pop(0)
                if predicate(message):
                    return message
                if message.startswith(("ERROR ","RESTORED ")):
                    raise RuntimeError("receiver ended unexpectedly: "+message)
                if message.startswith("N "):
                    return None
            if time.monotonic() >= until:
                return None
            self.poll()
            time.sleep(.001)

    def exchange(self, line, predicate, label):
        for attempt in range(self.retries):
            self.send(line)
            response = self.wait(predicate)
            if response is not None:
                return response
            self.event("RETRY",frame=label,attempt=attempt+1)
        raise TimeoutError("ACK timeout/retry limit: "+label)

    def abort(self):
        if self.boot_seen:
            return
        tag = "#S" if self.stage == "receiver" else "#B"
        self.exchange((tag+" "+self.sid+" X\n").encode(),lambda message:message.startswith("RESTORED "),"abort")


def start_receiver(link, target, size, expected_sha):
    # Do not retry the shell command: after a lost BOOT acknowledgement the
    # command could otherwise be consumed as input by a running bootstrap.
    link.send(bootstrap_command(link.sid,target,size,expected_sha),command=True)
    if link.wait(lambda message:message == "BOOT") is None:
        raise TimeoutError("bootstrap did not acknowledge startup; no payload sent")
    for sequence, line in enumerate(bootstrap_frames(link.sid)):
        fields = line.decode().split()
        wanted = "B %d %s" % (sequence,fields[3])
        link.exchange(line,lambda message,wanted=wanted:message == wanted,"bootstrap-"+str(sequence))
    link.event("BOOTSTRAP_EOF",compressed_sha256=digest(bootstrap_blob()),receiver_sha256=digest(RECEIVER_SOURCE.encode()))
    link.exchange(("#B "+link.sid+" E\n").encode(),lambda message:message == "READY","bootstrap-eof")
    link.stage = "receiver"


def send_file(link, source, size, expected_sha, progress=None):
    link.stage = "receiver"
    sent, sequence, hasher = 0, 0, hashlib.sha256()
    while sent < size:
        data = source.read(min(512,size-sent))
        if not data:
            raise ValueError("source shortened during upload")
        checksum = crc(data)
        frame = ("#S %s D %d %s %s\n" % (link.sid,sequence,checksum,base64.b64encode(data).decode())).encode()
        wanted = "A %d %s" % (sequence,checksum)
        link.exchange(frame,lambda message,wanted=wanted:message == wanted,"data-"+str(sequence))
        hasher.update(data);sent += len(data);sequence += 1
        if progress:
            progress(sent,size)
    if source.read(1) or hasher.hexdigest() != expected_sha:
        raise ValueError("source size/SHA changed; final publication refused")
    link.event("TX_EOF",size=size,sha256=expected_sha,frames=sequence)
    wanted = "DONE %d %s" % (size,expected_sha)
    response = link.exchange(("#S "+link.sid+" F\n").encode(),lambda message:message == wanted,"file-eof")
    link.event("BOARD_VERIFIED",receipt=response)
    link.exchange(("#S "+link.sid+" Q\n").encode(),lambda message:message == "RESTORED 0","terminal-restore")
    return dict(size=size,sha256=expected_sha,board_verified=True,terminal_restored=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port",default="COM13")
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--sha256",required=True)
    parser.add_argument("--target",required=True)
    parser.add_argument("--log",type=Path,required=True)
    parser.add_argument("--linux-shell-confirmed",action="store_true")
    parser.add_argument("--tx-delay",type=float,default=.015,help="seconds between TX slices; no per-slice flush")
    parser.add_argument("--tx-slice",type=int,default=128)
    parser.add_argument("--ack-timeout",type=float,default=2.)
    parser.add_argument("--retries",type=int,default=8)
    args = parser.parse_args(argv)
    port, link = None, None
    try:
        if not args.linux_shell_confirmed:
            raise ValueError("confirm an idle Linux shell with --linux-shell-confirmed; this tool cannot interrupt boot")
        validate_target(args.target)
        if not re.fullmatch(r"[0-9a-f]{64}",args.sha256):
            raise ValueError("exact lowercase source SHA256 required")
        if not (math.isfinite(args.tx_delay) and 0 <= args.tx_delay <= 1 and 16 <= args.tx_slice <= 128 and
                math.isfinite(args.ack_timeout) and .05 <= args.ack_timeout <= 30 and 1 <= args.retries <= 30):
            raise ValueError("invalid pacing/timeout/retry bound")
        if args.source.is_symlink() or not args.source.is_file():
            raise ValueError("source must be a regular file")
        size = args.source.stat().st_size
        if size > 67108864:
            raise ValueError("source exceeds 64 MiB")
        with args.source.open("rb") as source:
            hasher = hashlib.sha256()
            for data in iter(lambda:source.read(65536),b""):
                hasher.update(data)
            if hasher.hexdigest() != args.sha256:
                raise ValueError("source SHA256 mismatch; port not opened")
            source.seek(0)
            sid = secrets.token_hex(8)
            bootstrap_command(sid,args.target,size,args.sha256)
            with args.log.open("xb") as log:
                import serial
                port = serial.Serial(port=None,baudrate=115200,timeout=.05,write_timeout=5,
                                     xonxoff=False,rtscts=False,dsrdtr=False)
                port.dtr = False;port.rts = False;port.port = args.port
                port.open()
                link = Link(port,log,sid,timeout=args.ack_timeout,retries=args.retries,
                            tx_gap=args.tx_delay,tx_slice=args.tx_slice)
                link.event("START",session=sid,target=args.target,size=size,sha256=args.sha256,
                           tx_slice=args.tx_slice,tx_delay=args.tx_delay)
                started = time.monotonic()
                try:
                    # Passive boot-banner check only; never solicit a prompt or
                    # send a newline/control character to an unknown boot state.
                    until = time.monotonic()+.3
                    while time.monotonic() < until:
                        link.poll()
                    start_receiver(link,args.target,size,args.sha256)
                    last_report = [0]
                    def progress(sent,total):
                        if sent-last_report[0] >= 65536:
                            last_report[0] = sent
                            print(json.dumps(dict(acked_bytes=sent,total_bytes=total)),file=sys.stderr,flush=True)
                    result = send_file(link,source,size,args.sha256,progress)
                    result.update(target=args.target,elapsed_seconds=round(time.monotonic()-started,3),log=str(args.log))
                    link.event("SUCCESS",**result)
                    print(json.dumps(result,sort_keys=True),flush=True)
                except BaseException:
                    try:
                        link.abort()
                    except BaseException as cleanup:
                        link.event("RESTORE_UNCONFIRMED",reason=str(cleanup))
                    raise
    except (Exception,KeyboardInterrupt) as error:
        print("upload failed: "+str(error)+"; no success claimed; inspect "+str(args.log),file=sys.stderr)
        return 2
    finally:
        if port is not None:
            port.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
