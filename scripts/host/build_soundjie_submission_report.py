"""Build the competition report from authored prose, licensed references and project evidence."""
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle, Flowable, Image, KeepTogether)

ROOT = Path('E:/Soundjie_Work/d4')
ASSETS = ROOT/'artifacts/contest-report-final-20261102'
WORK = ROOT/'artifacts/contest-report-jury-20261102'
OUT = Path('C:/Users/32450/Desktop/presentation')
SOURCE = ROOT/'docs/submission/声界_作品技术报告_正式定稿.md'
PDF = OUT/'声界_作品技术报告_正式定稿.pdf'
FONT = Path('C:/Windows/Fonts')
W = 499.276
NAVY = colors.HexColor('#143A52')
TEAL = colors.HexColor('#087E8B')
INK = colors.HexColor('#253442')
GRAY = colors.HexColor('#5A6B78')
PALE = colors.HexColor('#EEF5F7')
LINE = colors.HexColor('#CBD9E1')

pdfmetrics.registerFont(TTFont('SJ', str(FONT/'msyh.ttc'), subfontIndex=0))
pdfmetrics.registerFont(TTFont('SJ-Bold', str(FONT/'msyhbd.ttc'), subfontIndex=0))
pdfmetrics.registerFontFamily('SJ', normal='SJ', bold='SJ-Bold', italic='SJ', boldItalic='SJ-Bold')
STYLES = {
    'body': ParagraphStyle('body',fontName='SJ',fontSize=10.4,leading=17.5,spaceAfter=9,textColor=INK,wordWrap='CJK'),
    'h1': ParagraphStyle('h1',fontName='SJ-Bold',fontSize=19,leading=27,spaceAfter=15,textColor=NAVY,keepWithNext=True),
    'h2': ParagraphStyle('h2',fontName='SJ-Bold',fontSize=11.8,leading=19,spaceBefore=7,spaceAfter=6,textColor=NAVY,keepWithNext=True),
    'cell': ParagraphStyle('cell',fontName='SJ',fontSize=9.2,leading=14.5,textColor=INK,wordWrap='CJK'),
    'thead': ParagraphStyle('thead',fontName='SJ-Bold',fontSize=9.2,leading=14.5,textColor=colors.white,wordWrap='CJK'),
    'caption': ParagraphStyle('caption',fontName='SJ',fontSize=8.4,leading=13,spaceBefore=6,spaceAfter=10,textColor=GRAY,alignment=TA_CENTER),
    'formula': ParagraphStyle('formula',fontName='SJ',fontSize=10.2,leading=19,leftIndent=13,borderPadding=7,spaceAfter=4,textColor=NAVY,backColor=PALE),
    'ref': ParagraphStyle('ref',fontName='SJ',fontSize=9.6,leading=17,spaceAfter=13,textColor=INK,wordWrap='CJK'),
}

def hsh(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paragraph(s, style='body'):
    s=escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    return Paragraph(s,STYLES[style])


def text(c, x, y, words, size=10, color=INK, bold=False, center=False):
    c.setFont('SJ-Bold' if bold else 'SJ',size)
    c.setFillColor(color)
    (c.drawCentredString if center else c.drawString)(x,y,words)


def box(c, x, y, w, h, lines, accent=False):
    c.setFillColor(NAVY if accent else PALE)
    c.setStrokeColor(NAVY if accent else LINE)
    c.roundRect(x,y,w,h,6,fill=1,stroke=1)
    for j,line in enumerate(lines):
        text(c,x+w/2,y+h/2+(len(lines)-1)*7-j*14-3,line,9.8,
             colors.white if accent else INK,bold=(j==0),center=True)


def arrow(c, points, color=TEAL):
    c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(1.2)
    for a,b in zip(points,points[1:]):c.line(*a,*b)
    x,y=points[-1];x0,y0=points[-2]
    ang=math.atan2(y-y0,x-x0)
    p=c.beginPath();p.moveTo(x,y)
    p.lineTo(x-5*math.cos(ang-.45),y-5*math.sin(ang-.45))
    p.lineTo(x-5*math.cos(ang+.45),y-5*math.sin(ang+.45));p.close()
    c.drawPath(p,fill=1,stroke=0)


class Diagram(Flowable):
    def __init__(self,kind,metrics):
        Flowable.__init__(self);self.width=W;self.kind=kind;self.metrics=metrics
        self.height={'architecture':290,'hardware':215,'doa':121,'enhance':145,
                     'software':105,'asr':128,'classifier':103,'confusion':275}[kind]
    def draw(self):
        c=self.canv;k=self.kind
        if k=='architecture':
            box(c,130,250,240,36,['M1  XVF3800四麦阵列','USB原始音频'])
            box(c,130,192,240,38,['M2  PS统一采集 / DDR缓冲'])
            arrow(c,[(250,250),(250,230)])
            box(c,0,113,240,51,['M3  PL阵列声学处理','定位 → 波束形成 → 降噪'],True)
            box(c,285,113,214,51,['M4  环境声分类分支','PS特征 → PL DPU'],True)
            arrow(c,[(210,192),(210,181),(120,181),(120,164)])
            arrow(c,[(290,192),(290,181),(392,181),(392,164)])
            text(c,17,173,'AXI DMA',8,GRAY)
            box(c,0,48,240,38,['M5  PS Vosk识别 / 文字解码'])
            box(c,285,48,214,38,['PS声音事件管理'])
            arrow(c,[(120,113),(120,86)])
            arrow(c,[(392,113),(392,86)])
            text(c,139,94,'增强音频',8,GRAY)
            box(c,105,0,290,31,['M6  HDMI：声音 / 字幕 / 方位 / 音量'])
            arrow(c,[(120,48),(120,39),(185,39),(185,31)])
            arrow(c,[(392,48),(392,39),(315,39),(315,31)])
            arrow(c,[(240,137),(262,137),(262,31)])
            text(c,266,98,'方位',8,GRAY)
        elif k=='hardware':
            c.translate(W*.07,0)
            c.scale(.86,.86)
            p=Path('C:/Users/32450/Downloads/IMG20261002143048.jpg')
            c.drawImage(ImageReader(str(p)),0,0,width=190,height=250,preserveAspectRatio=True,anchor='c',mask='auto')
            box(c,238,204,252,39,['四麦克风阵列','USB：音频与设备控制'])
            box(c,238,119,252,52,['MZ7100FA：PS＋PL','采集 / 声学计算 / 模型 / 显示'],True)
            arrow(c,[(364,204),(364,171)])
            box(c,238,49,116,40,['SD存储','启动 / 模型'])
            box(c,374,49,116,40,['HDMI显示器','声音与字幕'])
            arrow(c,[(296,89),(296,119)])
            arrow(c,[(432,119),(432,89)])
            text(c,238,15,'串口用于调试；运行任务在板端执行',8.5,GRAY)
        elif k=='doa':
            nodes=[['四路输入','分帧 / 加窗'],['FFT','互功率谱'],['PHAT归一化','IFFT'],['峰值搜索','亚采样插值'],['时差求角度','有效性 / 平滑']]
            width=89;gap=13
            for i,l in enumerate(nodes):
                x=i*(width+gap);box(c,x,52,width,49,l,accent=i in (1,2,3))
                if i<4:arrow(c,[(x+width,76),(x+width+gap,76)])
            text(c,W/2,23,'阵列坐标与通道映射 → 约束峰值搜索范围与方位求解',9,GRAY,center=True)
        elif k=='enhance':
            for i in range(4):
                text(c,0,120-i*26,f'MIC {i}',9,GRAY)
                box(c,48,108-i*26,118,23,['整数 / 分数延时'])
                arrow(c,[(166,119-i*26),(185,119-i*26),(185,78),(205,78)])
            box(c,205,54,87,48,['加权求和','目标方向'],True)
            box(c,319,54,180,48,['STFT / 维纳式增益','IFFT / 重叠相加'],True)
            arrow(c,[(292,78),(319,78)])
            text(c,319,27,'增强单路音频 → PS Vosk',9,GRAY)
        elif k=='software':
            labels=[['启动资源','设备初始化'],['USB采集','缓冲 / DMA'],['并行处理','声学 / 分类'],['结果组织','语音 / 事件'],['HDMI刷新','方向 / 字幕']]
            for i,l in enumerate(labels):
                x=i*102;box(c,x,42,90,45,l,accent=i==2)
                if i<4:arrow(c,[(x+90,64),(x+102,64)])
            arrow(c,[(453,42),(453,20),(147,20),(147,42)])
            text(c,230,7,'持续采集与异步结果更新',8.5,GRAY)
        elif k=='asr':
            labels=[['PL处理输出','单路增强PCM'],['PS缓冲','顺序送入'],['Vosk识别器','流式状态'],['字幕管理','临时 / 最终']]
            for i,l in enumerate(labels):
                x=i*127;box(c,x,57,115,49,l,accent=i==2)
                if i<3:arrow(c,[(x+115,81),(x+127,81)])
            text(c,W/2,29,'PartialResult更新当前句；Result确认已完成语句',9,GRAY,center=True)
            text(c,W/2,10,'增强音频在PL生成；识别与文字解码在PS执行',9,GRAY,center=True)
        elif k=='classifier':
            labels=[['声学特征','1×64×96'],['四组卷积池化','16/32/64/96'],['INT8 DPU','六类输出'],['分类阈值','提示 / 背景']]
            for i,l in enumerate(labels):
                x=i*127;box(c,x,35,115,49,l,accent=i==2)
                if i<3:arrow(c,[(x+115,59),(x+127,59)])
            text(c,W/2,10,'分类模型与Vosk语音模型分别运行，共同进入统一界面',9,GRAY,center=True)
        elif k=='confusion':
            mat=self.metrics['confusion_matrix'];names=['警笛','鸣笛','门铃','狗叫','人声','背景']
            x0,y0,cell=148,13,35
            text(c,252,259,'预测类别',9,GRAY,center=True)
            text(c,32,230,'真实类别',9,GRAY)
            for j,n in enumerate(names):text(c,x0+j*cell+cell/2,238,n,8.5,INK,center=True)
            for i,row in enumerate(mat):
                y=y0+(5-i)*cell
                c.setFillColor(INK);c.setFont('SJ',9);c.drawRightString(x0-9,y+12,names[i])
                for j,v in enumerate(row):
                    f=(v/sum(row))**.55 if v else 0
                    base=colors.HexColor('#EAF1F5');dark=NAVY
                    c.setFillColor(colors.Color(base.red*(1-f)+dark.red*f,base.green*(1-f)+dark.green*f,base.blue*(1-f)+dark.blue*f))
                    c.setStrokeColor(colors.white);c.rect(x0+j*cell,y,cell,cell,fill=1,stroke=1)
                    text(c,x0+j*cell+cell/2,y+12,str(v),10,colors.white if f>.55 else INK,center=True)


class Cover(Flowable):
    def __init__(self):
        super().__init__()
        self.width=W
        self.height=710
    def draw(self):
        c=self.canv
        text(c,0,680,'2026 全国大学生嵌入式芯片与系统设计竞赛',11,GRAY)
        text(c,0,653,'FPGA创新设计赛道 · AMD',12,TEAL,bold=True)
        c.setStrokeColor(TEAL);c.setLineWidth(3);c.line(0,620,70,620)
        text(c,0,547,'声界',54,NAVY,bold=True)
        text(c,0,501,'全离线空间听觉终端',25,NAVY,bold=True)
        text(c,0,460,'作品设计报告',17,GRAY)
        p=ROOT/'artifacts/contest-film-20261102/multi_full.jpg'
        c.drawImage(ImageReader(str(p)),0,140,width=W,height=W*9/16,mask='auto')
        for i,label in enumerate(['PL阵列声学','Vosk中文转写','声音分类','HDMI可视化']):
            x=i*127;c.setStrokeColor(LINE);c.setFillColor(PALE);c.roundRect(x,81,115,29,4,stroke=1,fill=1)
            text(c,x+57.5,91,label,10,TEAL,center=True)
        text(c,0,34,'让声音可见，让提示有方向。',12,NAVY)
        c.setFont('SJ',10);c.setFillColor(GRAY);c.drawRightString(W,34,'2026年11月')


def table(rows):
    cols=len(rows[0])
    if cols==2:widths=[W*.28,W*.72]
    elif cols==3:widths=[W*.23,W*.31,W*.46]
    elif cols==4:widths=[W*.28,W*.2,W*.26,W*.26]
    else:widths=[W/cols]*cols
    data=[[paragraph(s,'thead' if i==0 else 'cell') for s in r] for i,r in enumerate(rows)]
    t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[PALE,colors.white]),
                          ('LINEBELOW',(0,0),(-1,0),.8,NAVY),('LINEBELOW',(0,-1),(-1,-1),.6,LINE),
                          ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
                          ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    return t


class Report(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='h1':
            title=flowable.getPlainText();key=f'chapter-{self.page}'
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,level=0)


def page(c,doc):
    c.saveState()
    if doc.page>1:
        c.setStrokeColor(LINE);c.setLineWidth(.5);c.line(48,799,547,799)
        text(c,48,809,'声界  /  全离线空间听觉终端',8.2,GRAY)
        c.setFont('SJ',8.2);c.drawRightString(547,809,'作品设计报告')
    text(c,48,29,'FPGA创新设计赛道 · AMD',8,GRAY)
    c.setFont('SJ',8);c.drawRightString(547,29,f'{doc.page:02d}')
    c.restoreState()


def validate_vosk_resource_description(content):
    """Validate model-file size without requiring a runtime-memory statement."""
    model_label = r'模型(?:文件)?(?:大小|体积|规模)?[^\d。；;\n]{0,16}'
    assert re.search(model_label + r'45\s*MB(?![A-Za-z])', content, re.I), \
        'Vosk model files must be described as approximately 45 MB'
    sizes = re.findall(model_label + r'(\d+(?:\.\d+)?)\s*MB(?![A-Za-z])', content, re.I)
    assert all(float(size) == 45 for size in sizes), 'Conflicting model-file sizes'


def main():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    content=SOURCE.read_text(encoding='utf-8')
    banned=['待核','待补','初稿','待测','本稿','团队确认','以实际为准','尚未提供',
            '提交前','成稿核对','以团队最新功能说明','最终报告应','需要提供','应附',
            '建议','你们']
    for word in banned:assert word not in content,word
    validate_vosk_resource_description(content)
    report=json.loads((ASSETS/'classification-assessment.json').read_text(encoding='utf-8'))
    m=report['profiles'][report['recommended_profile']]['metrics']
    assert sum(sum(r) for r in m['confusion_matrix'])==314
    assert sum(m['confusion_matrix'][i][i] for i in range(6))==251
    assert round(m['accuracy']*100,2)==79.94 and round(m['macro_recall']*100,2)==73.58
    pages=content.split('<!-- page -->')
    story=[Cover(),PageBreak()]
    sources={
        'overall':ROOT/'artifacts/contest-film-20261102/dialogue_full.jpg',
        'siren':ASSETS/'siren-screen.jpg',
    }
    image_inputs=[ROOT/'artifacts/contest-film-20261102/multi_full.jpg',
                  Path('C:/Users/32450/Downloads/IMG20261002143048.jpg'),*sources.values()]
    for pi,block in enumerate(pages):
        lines=block.strip().splitlines();i=0
        while i<len(lines):
            line=lines[i].strip();i+=1
            if not line or line.startswith('# '):continue
            if line.startswith('## '):story.append(paragraph(line[3:],'h1'));continue
            if line.startswith('### '):story.append(paragraph(line[4:],'h2'));continue
            if line.startswith('|'):
                rows=[line.strip('|').split('|')]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    row=lines[i].strip().strip('|').split('|');i+=1
                    if all(re.fullmatch(r'[:\- ]+',v) for v in row):continue
                    rows.append(row)
                story.extend([table(rows),Spacer(1,8)]);continue
            if line.startswith('[[figure:'):
                key=line[len('[[figure:'):-2];story.append(Diagram(key,m));continue
            if line.startswith('[[photo:'):
                key=line[len('[[photo:'):-2];p=sources[key]
                w,h=ImageReader(str(p)).getSize();story.append(Image(str(p),width=W,height=W*h/w));continue
            if line=='[[table:metrics]]':
                rows=[['类别','正确 / 数量','召回率','精确率']]
                for label,ch in zip(report['labels'],['警笛','车辆鸣笛','门铃','狗叫','普通说话声','背景']):
                    v=m['per_class'][label];rows.append([ch,f"{v['correct']} / {v['support']}",f"{v['recall']*100:.2f}%",f"{v['precision']*100:.2f}%"])
                story.extend([table(rows),Spacer(1,6)]);continue
            if line.startswith('> '):story.append(paragraph(line[2:],'formula'));continue
            if re.match(r'^(图|表)\d',line):story.append(paragraph(line,'caption'));continue
            story.append(paragraph(line,'ref' if line.startswith('[') else 'body'))
        if pi<len(pages)-1:story.append(PageBreak())
    doc=Report(str(PDF),pagesize=(595.276,841.89),leftMargin=48,rightMargin=48,
               topMargin=59,bottomMargin=49,title='声界：全离线空间听觉终端 — 作品设计报告',
               author='声界项目团队',subject='FPGA创新设计赛道 AMD 作品设计报告')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    assert PDF.stat().st_size<50_000_000
    shutil.copyfile(SOURCE,OUT/'声界_作品技术报告_正式定稿.md')
    record=dict(pdf=str(PDF),bytes=PDF.stat().st_size,sha256=hsh(PDF),authored_pages=len(pages)+1,
                manuscript=str(SOURCE),manuscript_sha256=hsh(SOURCE),
                metrics_source=str(ASSETS/'classification-assessment.json'),metrics_sha256=hsh(ASSETS/'classification-assessment.json'),
                audience='competition_judges',model_files_mb=45,
                implementation_basis='User confirmed completed PL acoustics plus PS Vosk; model files are approximately 45 MB. Model-file size was supplied by the team, not measured by this report builder. Quantitative ASR/localization measurements were not asserted.',
                photos=[dict(path=str(p),sha256=hsh(p)) for p in image_inputs],board_accessed=False)
    (WORK/'report-manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=True))


if __name__=='__main__':main()
