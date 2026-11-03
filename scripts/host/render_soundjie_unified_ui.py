"""Render actual C++ UI frames from synthetic fixtures, without a board."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image

from tests.test_soundjie_unified_ui import snapshot, ATLAS
from tests.test_soundjie_hdmi_ui import wsl_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.binary.is_file():raise SystemExit('renderer binary missing')
    a.output.mkdir(parents=True,exist_ok=False)
    cases={
        '01-ordinary':snapshot(),
        '02-ambulance':snapshot(tv_matched=True,fine_valid=True),
        '03-background':snapshot(class_id=4,confidence=.9,class_scores=[.025]*4+[.9],
            doa_valid=False,doa_sector=None,doa_deg=None,doa_reason='no_speech',dbfs=-65),
        '04-caption-gap':snapshot(tv_matched=True,fine_valid=True,caption_text=''),
        '05-no-direction':snapshot(doa_valid=False,doa_sector=None,doa_deg=None,doa_reason='no_speech'),
        '06-fault':snapshot(session_state='fault',usb_ok=False,dpu_ok=False,
            doa_valid=False,doa_sector=None,doa_deg=None,doa_reason='unavailable'),
        '07-long-caption':snapshot(tv_matched=True,fine_valid=True,caption_text='打开百度地图'*10),
    }
    results=[]
    for name,state in cases.items():
        path=a.output/(name+'.json')
        path.write_text(json.dumps(state,ensure_ascii=False),encoding='utf-8')
        ppm=a.output/(name+'.ppm');report=a.output/(name+'-report.json')
        subprocess.run(['wsl','-d','Ubuntu','--',wsl_path(a.binary),
            '--snapshot',wsl_path(path),'--atlas',wsl_path(ATLAS),
            '--class-contract','soundjie-alert4bg-v2','--width','1280','--height','720',
            '--offscreen',wsl_path(ppm),'--report',wsl_path(report)],check=True,capture_output=True)
        png=a.output/(name+'.png')
        with Image.open(ppm) as image:image.save(png)
        results.append(dict(name=name,sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                            report=json.loads(report.read_text(encoding='utf-8'))))
    manifest=dict(synthetic_inputs=True,real_renderer=True,board_tested=False,
        binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        atlas_sha256=hashlib.sha256(ATLAS.read_bytes()).hexdigest(),cases=results)
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(a.output),frames=len(results),board_tested=False)))


if __name__=='__main__':main()
