#!/usr/bin/env python3
"""Bridge a serial console while automatically interrupting U-Boot autoboot.

The helper sends harmless space characters until it observes a U-Boot prompt.
It then becomes a transparent stdin/stdout serial bridge and records all bytes.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--interrupt-byte",
        type=lambda value: bytes.fromhex(value),
        default=b" ",
        help="single byte to transmit repeatedly, as two hexadecimal digits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.interrupt_byte) != 1:
        raise SystemExit("--interrupt-byte must decode to exactly one byte")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    finished = threading.Event()
    deadline = time.monotonic() + args.timeout

    with serial.Serial(args.port, args.baud, timeout=0.05) as uart, args.log.open("ab") as log:
        def reader() -> None:
            window = bytearray()
            while not finished.is_set():
                data = uart.read(4096)
                if not data:
                    continue
                log.write(data)
                log.flush()
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                window.extend(data)
                del window[:-4096]
                if b"Zynq>" in window or b"zynq>" in window:
                    stopped.set()

        def interrupter() -> None:
            while not stopped.is_set() and time.monotonic() < deadline:
                uart.write(args.interrupt_byte)
                uart.flush()
                time.sleep(0.05)

        threading.Thread(target=reader, daemon=True).start()
        threading.Thread(target=interrupter, daemon=True).start()
        previous_was_eol = False
        try:
            while True:
                data = sys.stdin.buffer.read(1)
                if not data:
                    break
                if data in (b"\r", b"\n"):
                    if previous_was_eol:
                        continue
                    data = b"\r"
                    previous_was_eol = True
                else:
                    previous_was_eol = False
                uart.write(data)
                uart.flush()
        except KeyboardInterrupt:
            pass
        finally:
            finished.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
