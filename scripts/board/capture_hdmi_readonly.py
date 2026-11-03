#!/usr/bin/env python3
"""Capture ARM32 Linux fb0 to a new PNG under /run; never changes display mode."""
import argparse
import fcntl
import mmap
import os
import struct
import zlib


def capture(output):
    if not os.path.realpath(output).startswith('/run/'):
        raise ValueError('output must be under /run')
    with open('/dev/fb0','rb',buffering=0) as f:
        var=fcntl.ioctl(f.fileno(),0x4600,bytes(160))
        fixed=fcntl.ioctl(f.fileno(),0x4602,bytes(80))
        w,h,vw,vh,x,y,bpp=struct.unpack_from('<7I',var)
        stride=struct.unpack_from('<I',fixed,44)[0] # ARM32 fb_fix_screeninfo
        if bpp!=32 or not 1<=w<=1920 or not 1<=h<=1080 or x+w>vw or y+h>vh or stride<vw*4 or stride*vh>16777216:
            raise ValueError('unsupported framebuffer layout')
        fields=[struct.unpack_from('<3I',var,n) for n in (32,44,56)]
        if any(length<1 or length>8 or offset+length>32 or msb for offset,length,msb in fields):
            raise ValueError('unsupported RGB fields')
        with mmap.mmap(f.fileno(),stride*vh,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ) as memory:
            data=memory[:]
    step=2; width=(w+step-1)//step; height=(h+step-1)//step
    raw=bytearray()
    for row in range(0,h,step):
        raw.append(0)
        for col in range(0,w,step):
            pixel=struct.unpack_from('<I',data,(y+row)*stride+(x+col)*4)[0]
            for offset,length,_ in fields:
                mask=(1<<length)-1
                raw.append(((pixel>>offset)&mask)*255//mask)
    def chunk(name,data):
        return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',width,height,8,2,0,0,0))
    png+=chunk(b'IDAT',zlib.compress(bytes(raw),1))+chunk(b'IEND',b'')
    with open(output,'xb') as f:f.write(png)
    print('FRAMEBUFFER_CAPTURE='+output+' bytes='+str(len(png)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    capture(p.parse_args().output)
