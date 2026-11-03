#!/usr/bin/env python3
"""Capture an MZ7100FA serial session and reliably interrupt U-Boot."""

import argparse
import base64
import shlex
import sys
import threading
import time
from pathlib import Path

import serial


AUTOBOOT_INTERRUPT_BYTE = b"\x03"


class UbootInterruptDetector:
    """Recognize the U-Boot banner once, including across read boundaries."""

    def __init__(self, marker=b"U-Boot"):
        self._marker = marker
        self._tail = b""
        self._triggered = False

    def feed(self, chunk):
        combined = self._tail + chunk
        self._tail = combined[-max(len(self._marker) - 1, 0) :]
        if not self._triggered and self._marker in combined:
            self._triggered = True
            return True
        return False


def encode_base64_lines(payload, width=76):
    encoded = base64.b64encode(payload)
    return [encoded[offset : offset + width] for offset in range(0, len(encoded), width)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM13")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interrupt-count", type=int, default=40)
    parser.add_argument("--interrupt-interval", type=float, default=0.025)
    return parser.parse_args()


def main():
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    detector = UbootInterruptDetector()
    stop = threading.Event()
    write_lock = threading.Lock()

    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.05,
        write_timeout=1,
        bytesize=8,
        parity="N",
        stopbits=1,
    ) as uart, args.log.open("ab", buffering=0) as log:

        def send(payload):
            with write_lock:
                uart.write(payload)
                uart.flush()

        def send_file(local_text, remote_text):
            local = Path(local_text)
            if not local.is_file():
                print(f"[host] local file does not exist: {local}", flush=True)
                return
            remote = shlex.quote(remote_text)
            lines = encode_base64_lines(local.read_bytes())
            total = len(lines)
            print(
                f"[host] Sending {local} ({local.stat().st_size} bytes) to {remote_text}.",
                flush=True,
            )
            send(f"stty -echo; base64 -d > {remote}\r".encode("ascii"))
            time.sleep(0.2)
            for start in range(0, total, 48):
                packet = b"\n".join(lines[start : start + 48]) + b"\n"
                send(packet)
                completed = min(start + 48, total)
                if completed == total or completed * 20 // total != start * 20 // total:
                    print(f"[host] File transfer: {completed * 100 // total}%", flush=True)
            # In canonical terminal mode, Ctrl-D on an empty line closes
            # base64's stdin without sending a byte into the decoded file.
            send(b"\x04")
            time.sleep(0.2)
            send(f"stty echo\rsha256sum {remote}\r".encode("ascii"))
            print("[host] Transfer queued; waiting for the board-side SHA-256.", flush=True)

        def reader():
            while not stop.is_set():
                chunk = uart.read(4096)
                if not chunk:
                    continue
                log.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                if detector.feed(chunk):
                    if args.interrupt_count > 0:
                        sys.stdout.write("\n[host] U-Boot detected; sending Ctrl-C interrupt keys.\n")
                    else:
                        sys.stdout.write("\n[host] U-Boot detected; receive-only startup, no interrupt keys sent.\n")
                    sys.stdout.flush()
                    for _ in range(args.interrupt_count):
                        send(AUTOBOOT_INTERRUPT_BYTE)
                        time.sleep(args.interrupt_interval)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        print(
            f"[host] Listening on {args.port} at {args.baud} 8N1; "
            f"log={args.log}. Enter U-Boot commands one per line; ::quit exits.",
            flush=True,
        )
        try:
            for line in sys.stdin:
                command = line.rstrip("\r\n")
                if command == "::quit":
                    break
                if command.startswith("::send "):
                    try:
                        parts = shlex.split(command)
                    except ValueError as error:
                        print(f"[host] invalid ::send command: {error}", flush=True)
                        continue
                    if len(parts) != 3:
                        print("[host] usage: ::send LOCAL_PATH REMOTE_PATH", flush=True)
                        continue
                    send_file(parts[1], parts[2])
                    continue
                send(command.encode("ascii") + b"\r")
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            thread.join(timeout=1)


if __name__ == "__main__":
    main()
