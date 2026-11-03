"""Create an editable report in the retained competition DOCX template."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import importlib.util
import json
import re

from lxml import etree as E
from PIL import Image
from reportlab.pdfgen import canvas
import pypdfium2 as pdfium

ROOT = Path('E:/Soundjie_Work/d4')
WORK = ROOT/'artifacts/contest-word-20261102'
TEMPLATE = Path('C:/Users/32450/Downloads/2658aa94-81d0-454c-b2ac-fc040ff2c01a.docx')
SOURCE = ROOT/'docs/submission/声界_作品技术报告_正式定稿.md'
OUTPUT = Path('C:/Users/32450/Desktop/presentation/声界_作品技术报告_可编辑公式版.docx')
EXPECTED_TEMPLATE_HASH = 'd15a2aa186f600cb9c234bffd9deff0c4e9b05c569643d8a81d536f66f424f0c'
NS = {
 'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
 'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
 'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
 'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
 'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
 'pic':'http://schemas.openxmlformats.org/drawingml/2006/picture',
 'rel':'http://schemas.openxmlformats.org/package/2006/relationships',
 'ct':'http://schemas.openxmlformats.org/package/2006/content-types',
}
PAGE_WIDTH = 8306


def sha(data): return hashlib.sha256(data).hexdigest()
def node(tag, text=None, **attrs):
    prefix, local = tag.split(':')
    n=E.Element('{'+NS[prefix]+'}'+local)
    for k,v in attrs.items():
        n.set('{'+NS[prefix]+'}'+k,str(v))
    if text is not None: n.text=str(text)
    return n
def add(parent,tag,text=None,**attrs):
    n=node(tag,text,**attrs);parent.append(n);return n
def write_xml(n): return E.tostring(n,xml_declaration=True,encoding='UTF-8',standalone=True)


def font(size=24,bold=False,east='宋体',latin='Times New Roman'):
    rp=node('w:rPr')
    add(rp,'w:rFonts',ascii=latin,hAnsi=latin,eastAsia=east,cs=latin)
    if bold:add(rp,'w:b')
    add(rp,'w:color',val='000000');add(rp,'w:sz',val=size);add(rp,'w:szCs',val=size)
    return rp


def mr(text,normal=False,size=28):
    r=node('m:r');mp=add(r,'m:rPr')
    add(mp,'m:sty',val='p' if normal else 'i')
    r.append(font(size,latin='Cambria Math'))
    t=add(r,'m:t',text);t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    return r
def slot(parent,tag,items):
    s=add(parent,tag)
    for x in items if isinstance(items,list) else [items]:s.append(x)
    return s
def sub(base,idx):
    s=node('m:sSub');slot(s,'m:e',mr(base));slot(s,'m:sub',mr(idx));return s
def sup(base,power):
    s=node('m:sSup');slot(s,'m:e',base if not isinstance(base,str) else mr(base));slot(s,'m:sup',mr(power));return s
def subsup(base,idx,power):
    s=node('m:sSubSup');slot(s,'m:e',mr(base));slot(s,'m:sub',mr(idx));slot(s,'m:sup',mr(power));return s
def frac(num,den):
    f=node('m:f');slot(f,'m:num',num);slot(f,'m:den',den);return f
def delim(items,left='[',right=']'):
    d=node('m:d');pr=add(d,'m:dPr');add(pr,'m:begChr',val=left);add(pr,'m:endChr',val=right)
    slot(d,'m:e',items);return d
def seq(base,idx,arg):return [sub(base,idx),delim(mr(arg))]
def summation(lo,hi,expr):
    n=node('m:nary');pr=add(n,'m:naryPr');add(pr,'m:chr',val='∑');add(pr,'m:limLoc',val='undOvr');add(pr,'m:grow',val=0)
    slot(n,'m:sub',mr(lo));slot(n,'m:sup',mr(hi));slot(n,'m:e',expr);return n
def tilde():
    a=node('m:acc');pr=add(a,'m:accPr');add(pr,'m:chr',val='̃');slot(a,'m:e',mr('H'));return a


def equations():
    eq=[]
    eq.append(seq('G','ij','k')+[mr('='),sub('X','i'),delim(mr('k')),subsup('X','j','∗'),delim(mr('k'))])
    eq.append(seq('C','ij','k')+[mr('='),frac(seq('G','ij','k'),[delim(seq('G','ij','k'),'|','|'),mr('+'),mr('ε')])])
    eq.append(seq('r','ij','ℓ')+[mr('='),sup(mr('ℱ',normal=True),'−1'),delim(seq('C','ij','k'),'{','}')])
    eq.append([sub('τ','ij'),mr('='),frac([sub('ℓ','ij'),mr('+'),sub('δ','ij')],sub('f','s'))])
    inner=[sub('h','i'),delim([mr('q;'),sub('d','i')]),sub('x','i'),delim([mr('n−'),sub('D','i'),mr('−q')])]
    eq.append([mr('y'),delim(mr('n')),mr('='),summation('i=0','3',[sub('w','i'),summation('q=0','L−1',inner)])])
    eq.append([mr('H'),delim(mr('k')),mr('='),frac(seq('P','s','k'),seq('P','s','k')+[mr('+')]+seq('P','v','k')+[mr('+'),mr('ε')])])
    eq.append([mr('Y'),delim(mr('k')),mr('='),tilde(),delim(mr('k')),mr('X'),delim(mr('k'))])
    return eq


TOKENS = {
 'Xi':lambda:sub('X','i'),'Xj':lambda:sub('X','j'),'eps':lambda:mr('ε'),
 'peak_ij':lambda:sub('ℓ','ij'),'delta_ij':lambda:sub('δ','ij'),'fs':lambda:sub('f','s'),
 'D_i':lambda:sub('D','i'),'d_i':lambda:sub('d','i'),'w_i':lambda:sub('w','i'),
 'Psignal':lambda:sub('P','s'),'Pnoise':lambda:sub('P','v'),'Hsmooth':tilde,
 'FIR_COEFF':lambda:[sub('h','i'),delim([mr('q;'),sub('d','i')])],
 'FILTER_LENGTH':lambda:mr('L'),
}
TOKEN_RX=re.compile(r'(?<![A-Za-z])('+ '|'.join(TOKENS)+r')(?![A-Za-z])')


def ppr(style='SJBody',keep=None):
    pr=node('w:pPr');add(pr,'w:pStyle',val=style)
    if keep is not None:add(pr,'w:keepNext',val=1 if keep else 0)
    return pr
def append_plain(p,text,bold=False):
    for j,segment in enumerate(re.split(r'(\*\*[^*]+\*\*)',text)):
        b=bold or segment.startswith('**');segment=segment.strip('*') if segment.startswith('**') else segment
        pos=0
        for match in TOKEN_RX.finditer(segment):
            if match.start()>pos:append_run(p,segment[pos:match.start()],b)
            om=node('m:oMath');elements=TOKENS[match.group()]();elements=elements if isinstance(elements,list) else [elements]
            for e in elements:
                for sz in e.xpath('.//w:sz|.//w:szCs',namespaces=NS):sz.set('{'+NS['w']+'}val','24')
                om.append(e)
            p.append(om);pos=match.end()
        if pos<len(segment):append_run(p,segment[pos:],b)
def append_run(p,text,bold=False):
    if not text:return
    r=add(p,'w:r')
    if bold:r.append(font(bold=True))
    t=add(r,'w:t',text);t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
def para(text='',style='SJBody',keep=None):
    p=node('w:p');p.append(ppr(style,keep));append_plain(p,text);return p
def display_math(items):
    p=node('w:p');p.append(ppr('SJEquation'))
    ompara=add(p,'m:oMathPara');pr=add(ompara,'m:oMathParaPr');add(pr,'m:jc',val='center')
    om=add(ompara,'m:oMath')
    for e in items:om.append(e)
    return p


def styles_append(styles):
    def st(sid,name,size=24,bold=False,align='both',before=0,after=100,outline=None,first=480,line=360):
        s=node('w:style',type='paragraph',styleId=sid);add(s,'w:name',val=name);add(s,'w:basedOn',val='1');add(s,'w:next',val='SJBody');add(s,'w:qFormat')
        pr=add(s,'w:pPr');add(pr,'w:widowControl');add(pr,'w:snapToGrid',val=0)
        add(pr,'w:spacing',before=before,after=after,line=line,lineRule='auto')
        add(pr,'w:ind',left=0,right=0,firstLine=first);add(pr,'w:jc',val=align)
        if outline is not None:
            add(pr,'w:keepNext');add(pr,'w:keepLines');add(pr,'w:outlineLvl',val=outline)
        s.append(font(size,bold));styles.append(s);return s
    st('SJBody','声界正文')
    st('SJTitle','Title',36,False,'center',0,200,first=0)
    st('SJSubtitle','声界副标题',24,False,'center',0,180,first=0)
    st('SJChapter','声界一级标题',28,True,'left',120,160,0,0)
    st('SJSection','声界二级标题',24,True,'left',180,100,1,0)
    st('SJSubsection','声界三级标题',24,True,'left',140,80,2,0)
    st('SJMinor','声界四级标题',24,True,'left',100,60,3,0)
    st('SJCaption','声界图表题',21,False,'center',70,130,first=0,line=280)
    st('SJFigure','声界图片',24,False,'center',60,0,first=0,line=240)
    st('SJEquation','声界公式',28,False,'center',100,100,first=0,line=300)
    st('SJCell','声界表格',22,False,'left',0,0,first=0,line=260)
    ref=st('SJReference','声界参考文献',24,False,'left',0,140,first=0,line=320)
    add(ref.find('w:pPr',NS),'w:keepLines')


def table(rows):
    cols=len(rows[0]);ratios={2:[.28,.72],3:[.23,.31,.46],4:[.28,.20,.26,.26]}.get(cols,[1/cols]*cols)
    widths=[round(PAGE_WIDTH*x) for x in ratios];widths[-1]+=PAGE_WIDTH-sum(widths)
    t=node('w:tbl');pr=add(t,'w:tblPr');add(pr,'w:tblStyle',val='7');add(pr,'w:tblW',w=PAGE_WIDTH,type='dxa');add(pr,'w:jc',val='center');add(pr,'w:tblLayout',type='fixed')
    margins=add(pr,'w:tblCellMar')
    for s,v in [('top',90),('bottom',90),('left',100),('right',100)]:add(margins,'w:'+s,w=v,type='dxa')
    borders=add(pr,'w:tblBorders')
    for s in ['top','left','bottom','right','insideH','insideV']:add(borders,'w:'+s,val='single',sz=5,color='D9D9D9')
    grid=add(t,'w:tblGrid')
    for w in widths:add(grid,'w:gridCol',w=w)
    for i,row in enumerate(rows):
        tr=add(t,'w:tr');trpr=add(tr,'w:trPr');add(trpr,'w:cantSplit')
        if i==0:add(trpr,'w:tblHeader')
        for j,value in enumerate(row):
            tc=add(tr,'w:tc');tcp=add(tc,'w:tcPr');add(tcp,'w:tcW',w=widths[j],type='dxa');add(tcp,'w:vAlign',val='center')
            add(tcp,'w:shd',val='clear',fill='EAEAEA' if i==0 else ('F7F7F7' if i%2==0 else 'FFFFFF'))
            p=para(value,'SJCell',keep=i==0)
            if i==0:
                for r in p.findall('w:r',NS):
                    rp=r.find('w:rPr',NS)
                    if rp is None:rp=node('w:rPr');r.insert(0,rp)
                    add(rp,'w:b')
            if cols==4 and j>0:add(p.find('w:pPr',NS),'w:jc',val='center')
            tc.append(p)
    return t


def diagram_assets():
    spec=importlib.util.spec_from_file_location('pdf_report',ROOT/'scripts/host/build_soundjie_submission_report.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assessment=json.loads((mod.ASSETS/'classification-assessment.json').read_text(encoding='utf8'))
    metrics=assessment['profiles'][assessment['recommended_profile']]['metrics']
    out={};folder=WORK/'figures';folder.mkdir(exist_ok=True,parents=True)
    for key in ['architecture','hardware','doa','enhance','software','asr','classifier','confusion']:
        item=mod.Diagram(key,metrics);p=folder/f'{key}.pdf'
        c=canvas.Canvas(str(p),pagesize=(mod.W,item.height));item.drawOn(c,0,0);c.save()
        d=pdfium.PdfDocument(str(p));im=d[0].render(scale=4).to_pil();dest=folder/f'{key}.png';im.save(dest);out[key]=dest;d.close()
    return out,assessment,metrics


def main():
    assert sha(TEMPLATE.read_bytes())==EXPECTED_TEMPLATE_HASH
    assert (WORK/'artifact.md').exists()
    figures,assessment,metrics=diagram_assets()
    with ZipFile(TEMPLATE) as z:
        parts={n:z.read(n) for n in z.namelist()}
    doc=E.fromstring(parts['word/document.xml']);body=doc.find('w:body',NS);section=deepcopy(body.find('w:sectPr',NS))
    body.clear()
    styles=E.fromstring(parts['word/styles.xml']);styles_append(styles)
    rels=E.fromstring(parts['word/_rels/document.xml.rels'])
    contenttypes=E.fromstring(parts['[Content_Types].xml'])
    addmedia={};picture_index=0

    def picture(path,key):
        nonlocal picture_index
        picture_index+=1;rid=f'rIdSJ{picture_index}';ext=path.suffix.lower().lstrip('.');filename=f'soundjie_{picture_index}.{ext}'
        media_name='word/media/'+filename;addmedia[media_name]=path.read_bytes()
        relationship=E.SubElement(rels,'{'+NS['rel']+'}Relationship');relationship.set('Id',rid);relationship.set('Type',NS['r']+'/image');relationship.set('Target','media/'+filename)
        if not any(n.get('Extension')==ext for n in contenttypes):
            n=E.SubElement(contenttypes,'{'+NS['ct']+'}Default');n.set('Extension',ext);n.set('ContentType','image/png' if ext=='png' else 'image/jpeg')
        w,h=Image.open(path).size
        width=PAGE_WIDTH*635;height=round(width*h/w)
        p=para(style='SJFigure',keep=True);r=add(p,'w:r');dr=add(r,'w:drawing');inline=add(dr,'wp:inline',distT=0,distB=0,distL=0,distR=0)
        # DrawingML attributes on these elements are unqualified.
        for a in list(inline.attrib):v=inline.attrib.pop(a);inline.set(E.QName(a).localname,v)
        extent=E.SubElement(inline,'{'+NS['wp']+'}extent',cx=str(width),cy=str(height))
        E.SubElement(inline,'{'+NS['wp']+'}docPr',id=str(10000+picture_index),name='声界图 '+str(picture_index),descr=key)
        frame=E.SubElement(inline,'{'+NS['wp']+'}cNvGraphicFramePr');E.SubElement(frame,'{'+NS['a']+'}graphicFrameLocks',noChangeAspect='1')
        g=E.SubElement(inline,'{'+NS['a']+'}graphic');gd=E.SubElement(g,'{'+NS['a']+'}graphicData',uri=NS['pic']);pic=E.SubElement(gd,'{'+NS['pic']+'}pic')
        nv=E.SubElement(pic,'{'+NS['pic']+'}nvPicPr');E.SubElement(nv,'{'+NS['pic']+'}cNvPr',id='0',name=filename);E.SubElement(nv,'{'+NS['pic']+'}cNvPicPr')
        fill=E.SubElement(pic,'{'+NS['pic']+'}blipFill');blip=E.SubElement(fill,'{'+NS['a']+'}blip');blip.set('{'+NS['r']+'}embed',rid)
        stretch=E.SubElement(fill,'{'+NS['a']+'}stretch');E.SubElement(stretch,'{'+NS['a']+'}fillRect')
        sp=E.SubElement(pic,'{'+NS['pic']+'}spPr');xf=E.SubElement(sp,'{'+NS['a']+'}xfrm');E.SubElement(xf,'{'+NS['a']+'}off',x='0',y='0');E.SubElement(xf,'{'+NS['a']+'}ext',cx=str(width),cy=str(height))
        geom=E.SubElement(sp,'{'+NS['a']+'}prstGeom',prst='rect');E.SubElement(geom,'{'+NS['a']+'}avLst')
        body.append(p)

    body.append(para('声界 全离线空间听觉终端','SJTitle'))
    body.append(para('作品设计报告','SJSubtitle'))
    eqs=equations();eq_index=0;mode='abstract';skip_guide=False
    photo_paths={'overall':ROOT/'artifacts/contest-film-20261102/dialogue_full.jpg','siren':ROOT/'artifacts/contest-report-final-20261102/siren-screen.jpg'}
    hmaps={
      '第二部分　硬件系统':('2.2 硬件系统介绍','SJSection','hardware'),
      '第二部分　PL声源定位':('2.2.4 PL声源定位','SJSubsection','doa'),
      '第二部分　PL波束形成与降噪':('2.2.5 PL波束形成与降噪','SJSubsection','enhance'),
      '第二部分　软件结构与模块接口':('2.3 软件系统介绍','SJSection','software'),
      '第二部分　离线中文语音转写':('2.3.4 离线中文语音转写','SJSubsection','asr'),
      '第二部分　环境声分类与提醒':('2.3.5 环境声分类与提醒','SJSubsection','classifier'),
      '第三部分　HDMI交互展示':('3.2.4 HDMI交互展示','SJSubsection','ui'),
    }
    lines=SOURCE.read_text(encoding='utf8').splitlines();i=0
    while i<len(lines):
        s=lines[i].strip();i+=1
        if s=='<!-- page -->':skip_guide=False;continue
        if not s or s.startswith('# '):continue
        if s=='### 报告内容':skip_guide=True;continue
        if skip_guide:continue
        if s.startswith('## '):
            title=s[3:]
            if title in ['第三部分　环境声模型验证','第三部分　场景测试与系统运行']:
                mode='results';continue
            if title in hmaps:
                title,sty,mode=hmaps[title];body.append(para(title,sty));continue
            p=para(title,'SJChapter')
            mode='part';body.append(p);continue
        if s.startswith('### '):
            title=s[4:]
            sty='SJMinor' if mode in ['doa','enhance','asr','classifier','ui'] else 'SJSection'
            if re.match(r'\d+\.\d+\.\d+',title):sty='SJSubsection'
            if title=='数据通路与控制通路':title='2.3.3 '+title;sty='SJSubsection'
            body.append(para(title,sty));continue
        if s.startswith('|'):
            rows=[s.strip('|').split('|')]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=lines[i].strip().strip('|').split('|');i+=1
                if all(re.fullmatch(r'[:\- ]+',v) for v in row):continue
                rows.append(row)
            body.append(table(rows));continue
        if s.startswith('[[figure:'):
            key=s[len('[[figure:'):-2];picture(figures[key],key);continue
        if s.startswith('[[photo:'):
            key=s[len('[[photo:'):-2];picture(photo_paths[key],key);continue
        if s=='[[table:metrics]]':
            rows=[['类别','正确 / 数量','召回率','精确率']]
            for label,ch in zip(assessment['labels'],['警笛','车辆鸣笛','门铃','狗叫','普通说话声','背景']):
                v=metrics['per_class'][label];rows.append([ch,f"{v['correct']} / {v['support']}",f"{v['recall']*100:.2f}%",f"{v['precision']*100:.2f}%"])
            body.append(table(rows));continue
        if s.startswith('> '):
            body.append(display_math(eqs[eq_index]));eq_index+=1;continue
        if re.match(r'^(图|表)\d',s):
            if len(body) and E.QName(body[-1]).localname=='tbl':
                for p in body[-1].findall('w:tr',NS)[-1].findall('.//w:p',NS):add(p.find('w:pPr',NS),'w:keepNext')
            body.append(para(s,'SJCaption'));continue
        if s.startswith('D_i和d_i分别'):
            s='D_i和d_i分别表示整数延迟与分数延迟，w_i为通道权重，FIR_COEFF为对应分数延迟的FIR系数，FILTER_LENGTH为滤波器抽头数。经过时延补偿，目标方向声音在求和时保持相干叠加，使后续语音处理聚焦于当前关注方向。'
        body.append(para(s,'SJReference' if re.match(r'^\[\d+\]',s) else 'SJBody'))
    assert eq_index==7
    body.append(section)
    settings=E.fromstring(parts['word/settings.xml']);add(settings,'w:updateFields',val='true')
    modified={
      'word/document.xml':write_xml(doc),'word/styles.xml':write_xml(styles),
      'word/_rels/document.xml.rels':write_xml(rels),'[Content_Types].xml':write_xml(contenttypes),
      'word/settings.xml':write_xml(settings),
    }
    if 'docProps/core.xml' in parts:
        core=E.fromstring(parts['docProps/core.xml']);dc='http://purl.org/dc/elements/1.1/';cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties'
        for tag,value in [(dc+'title','声界 全离线空间听觉终端 作品设计报告'),(dc+'creator','声界项目团队'),(cp+'/lastModifiedBy','声界项目团队')]:
            namespace,local=tag.rsplit('/',1) if tag.startswith(cp+'/') else (dc,tag[len(dc):])
            q='{'+namespace+'}'+local;n=core.find(q)
            if n is None:n=E.SubElement(core,q)
            n.text=value
        modified['docProps/core.xml']=write_xml(core)
    with ZipFile(OUTPUT,'w',ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,modified.get(name,data))
        for name,data in addmedia.items():z.writestr(name,data)
    with ZipFile(OUTPUT) as z:
        for name,data in parts.items():
            if name not in modified:assert z.read(name)==data,name
        assert z.read('word/media/image1.png')==parts['word/media/image1.png']
        xml=E.fromstring(z.read('word/document.xml'))
        assert len(xml.xpath('//m:oMathPara',namespaces=NS))==7
        wt=''.join(xml.xpath('//w:t/text()',namespaces=NS))
        for bad in ['sum_i','delay(','tau_ij','peak_ij','Gij[','Hsmooth[','Psignal[','FIR_COEFF','FILTER_LENGTH']:assert bad not in wt,bad
    assert sha(TEMPLATE.read_bytes())==EXPECTED_TEMPLATE_HASH
    record={'output':str(OUTPUT),'bytes':OUTPUT.stat().st_size,'sha256':sha(OUTPUT.read_bytes()),
      'reference':str(TEMPLATE),'template_sha256':EXPECTED_TEMPLATE_HASH,'source':str(SOURCE),'source_sha256':sha(SOURCE.read_bytes()),
      'native_display_equations':7,'native_math_objects':len(doc.xpath('//m:oMath',namespaces=NS)),
      'header_image_sha256':sha(parts['word/media/image1.png']),'preserved_parts':{n:sha(v) for n,v in parts.items() if n not in modified},
      'changed_original_parts':list(modified),'added_media':list(addmedia),'section_xml_preserved':True}
    (WORK/'build-manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in record.items() if k not in ['preserved_parts','added_media']},ensure_ascii=True))


if __name__=='__main__':main()
