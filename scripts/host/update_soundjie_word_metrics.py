"""Add measured DPU/HDMI platform metrics to the user's edited DOCX.

Preserve the source, every existing body element, and all other package parts.
The only edited existing element is the requested performance summary table.
"""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import re
from lxml import etree as E

ROOT = Path('E:/Soundjie_Work/d4')
WORK = Path('F:/MLK/DPU_ZYNQ7000/artifacts/contest_metrics_20261102')
SOURCE = Path('C:/Users/32450/Desktop/presentation/声界_作品技术报告_可编辑公式版.docx')
OUTPUT = SOURCE.with_name('声界_作品技术报告_性能指标补充版.docx')
BUILD = ROOT/'artifacts/pl_rebuild_20261024_emmc_a'
RUN = BUILD/'sources/ch24/CH24_LCD_Tourch/02_fpga/fpga/system/system.runs/impl_1'
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W = '{'+NS['w']+'}'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def add(parent, tag, **attrs):
    n=E.SubElement(parent,W+tag)
    for k,v in attrs.items():n.set(W+k,str(v))
    return n


def plain(n):
    return ''.join(n.xpath('.//w:t/text()',namespaces=NS))


def extract_static():
    paths={'utilization':BUILD/'result_a/utilization.rpt',
           'timing':RUN/'system_wrapper_timing_summary_routed.rpt',
           'power':RUN/'system_wrapper_power_routed.rpt',
           'bitstream':BUILD/'result_a/soundjie_d4_combined.bit'}
    contents={k:p.read_bytes() for k,p in paths.items()}
    assert sha(contents['bitstream'])=='6bc37bf341c549c1c4bfd306f7a3a4275ed07411eaa81fe6992897475af39118'
    util=contents['utilization'].decode();timing=contents['timing'].decode();power=contents['power'].decode()
    resources=[]
    for name,label in [('Slice LUTs','LUT'),('Slice Registers','触发器 FF'),('Block RAM Tile','BRAM（36 Kb等效块）'),('DSPs','DSP48E1')]:
        m=re.search(r'\|\s*'+re.escape(name)+r'\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',util)
        assert m,name
        used,total,pct=m.groups();resources.append([label,used,total,pct+'%'])
    assert '0.254        0.000' in timing and '0.029        0.000' in timing
    for value in ['142.857','74.286','371.429','150.000','300.000']:assert value in timing
    assert '5.136' in power and '4.851' in power and '0.285' in power and 'Low' in power
    return resources,{k:{'path':str(paths[k]),'sha256':sha(v)} for k,v in contents.items()}


def main():
    source_manifest=json.loads((WORK/'source-parts.json').read_text(encoding='utf8'))
    source_bytes=SOURCE.read_bytes()
    assert sha(source_bytes)==source_manifest['sha256'],'User source changed; re-read before editing.'
    metrics=json.loads((WORK/'live-metrics.json').read_text(encoding='utf8'))
    evidence=json.loads((WORK/'current-board-evidence.json').read_text(encoding='utf8'))
    resources,static_sources=extract_static()
    assert metrics['window_count']==436 and metrics['bad_windows']==0
    assert 'B1152F' in evidence['dpu_info'] and '150 MHz' in evidence['dpu_info']
    with ZipFile(SOURCE) as z:parts={n:z.read(n) for n in z.namelist()}
    doc=E.fromstring(parts['word/document.xml']);body=doc.find('w:body',NS)
    original_nodes=[E.tostring(n) for n in body]
    style_xml=E.fromstring(parts['word/styles.xml'])
    styles={n.find('w:name',NS).get(W+'val'):n.get(W+'styleId') for n in style_xml.findall('w:style',NS)}

    def paragraph(text,role='声界正文'):
        p=E.Element(W+'p');pr=add(p,'pPr');add(pr,'pStyle',val=styles[role])
        if role in ('声界二级标题','声界三级标题'):add(pr,'keepNext');add(pr,'keepLines')
        r=add(p,'r');t=add(r,'t');t.text=text;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
        return p

    def table(rows,ratios):
        width=8306;ws=[round(width*r) for r in ratios];ws[-1]+=width-sum(ws)
        t=E.Element(W+'tbl');pr=add(t,'tblPr');add(pr,'tblW',w=width,type='dxa');add(pr,'jc',val='center');add(pr,'tblLayout',type='fixed')
        mar=add(pr,'tblCellMar')
        for k,v in [('top',90),('bottom',90),('left',100),('right',100)]:add(mar,k,w=v,type='dxa')
        bd=add(pr,'tblBorders')
        for k in ['top','left','bottom','right','insideH','insideV']:add(bd,k,val='single',sz=5,color='D9D9D9')
        g=add(t,'tblGrid')
        for v in ws:add(g,'gridCol',w=v)
        for i,row in enumerate(rows):
            tr=add(t,'tr');rp=add(tr,'trPr');add(rp,'cantSplit')
            if i==0:add(rp,'tblHeader')
            for j,value in enumerate(row):
                c=add(tr,'tc');cp=add(c,'tcPr');add(cp,'tcW',w=ws[j],type='dxa');add(cp,'vAlign',val='center')
                add(cp,'shd',val='clear',fill='EAEAEA' if i==0 else ('F7F7F7' if i%2==0 else 'FFFFFF'))
                p=paragraph(str(value),'声界表格')
                if i==0 or (len(rows)<=7 and i<len(rows)-1):
                    add(p.find('w:pPr',NS),'keepNext')
                if i==0:
                    for r in p.findall('w:r',NS):
                        rpr=E.Element(W+'rPr');add(rpr,'b');r.insert(0,rpr)
                c.append(p)
        return t

    heading=next(n for n in body if plain(n)=='1.4 主要性能指标')
    summary=heading.getnext();assert E.QName(summary).localname=='tbl'
    original_summary_rows=[E.tostring(n) for n in summary.findall('w:tr',NS)]
    extra=[
        ['PL资源占用（DPU＋HDMI平台）','LUT 12.86%；FF 13.94%；BRAM 16.89%；DSP 10.45%'],
        ['PL工作时钟','总线142.857 MHz；DPU 150/300 MHz；HDMI像素74.286 MHz'],
        ['已约束路径时序','WNS 0.254 ns；WHS 0.029 ns；TNS、THS均为0'],
        ['DPU配置与实测','单核B1152F；436次调用中位数0.653 ms，95百分位1.003 ms'],
        ['声音结果更新','约2次/秒；接收窗口后软件处理耗时中位数440 ms']
    ]
    template_row=summary.findall('w:tr',NS)[-1]
    for values in extra:
        row=deepcopy(template_row)
        for cell,value in zip(row.findall('w:tc',NS),values):
            for child in list(cell):
                if E.QName(child).localname!='tcPr':cell.remove(child)
            cell.append(paragraph(value,'声界表格'))
        summary.append(row)
    note=paragraph('表中新增硬件指标对应已部署的DPU＋HDMI基础平台；资源与时序由布局布线报告提取，运行计时来自2026年11月2日板端记录。详细统计范围见3.8节。')
    summary.addnext(note)

    blocks=[]
    def p(text,role='声界正文'):blocks.append(paragraph(text,role))
    def tab(rows,ratios):
        if blocks and E.QName(blocks[-1]).localname=='tbl':
            gap=paragraph('')
            pr=gap.find('w:pPr',NS)
            add(pr,'spacing',before=0,after=0,line=80,lineRule='exact')
            add(pr,'keepNext')
            blocks.append(gap)
        blocks.append(table(rows,ratios))
    p('3.8 FPGA实现与板端运行性能','声界二级标题')
    p('本节对已部署的DPU＋HDMI硬件平台进行量化分析。硬件实现采用Vivado 2017.4，目标器件为XC7Z100-2FFG900，统计对象为SJEM1版本的布局布线网表，包含DPU、AXI互连、VDMA、HDMI输出和相关接口。板端启动读出的器件与构建日期同该版本一致；以下资源统计针对这一平台，不合并其他版本的声学模块。')
    p('3.8.1 可编程逻辑资源占用','声界三级标题')
    tab([['资源类型','已用数量','可用总量','占用率']]+resources,[.34,.20,.22,.24])
    p('BRAM按36 Kb等效块统计，116个RAMB36与23个RAMB18合计为127.5块。LUT、FF、BRAM与DSP的占用率分别为12.86%、13.94%、16.89%与10.45%，反映了该DPU与显示平台的实际逻辑、存储和乘加资源使用情况。')
    p('3.8.2 时钟配置与时序分析','声界三级标题')
    tab([['时钟域','工作频率','用途'],['PS输出与AXI控制域','142.857 MHz','总线控制、VDMA与显示数据通路'],['DPU主时钟','150.000 MHz','神经网络推理计算'],['DPU双倍频时钟','300.000 MHz','DPU内部计算时钟'],['HDMI像素时钟','74.286 MHz','1280×720视频像素输出'],['HDMI串行化时钟','371.429 MHz','像素时钟的5倍频串行化']],[.31,.25,.44])
    tab([['时序指标','布局布线结果'],['最差建立时间裕量 WNS','＋0.254 ns'],['建立时间总负裕量 TNS','0 ns；失败端点0'],['最差保持时间裕量 WHS','＋0.029 ns'],['保持时间总负裕量 THS','0 ns；失败端点0'],['最差脉宽裕量 WPWS','＋1.267 ns']],[.56,.44])
    p('已约束路径满足设定时序要求，建立时间与保持时间均无负裕量。上述结果适用于报告中的已约束路径；I/O检查另有5个未约束端点、5个未设置输入延迟的端口和31个未设置输出延迟的端口，主要涉及eMMC与保留显示接口。')
    p('3.8.3 DPU配置与推理耗时','声界三级标题')
    p('板端DPU信息读取结果为单核B1152F、目标版本v1.4.0、运行频率150 MHz，支持平均池化、LeakyReLU/ReLU6与逐通道卷积。环境声模型采用INT8量化输入与计算，分类网络输入张量为1×64×96。')
    p('计时在板端使用单调时钟包围dpuRunTask调用，覆盖任务提交至返回，不包含音频窗口积累、特征计算、进程启动与HDMI刷新。本次连续运行保留436个推理窗口，其统计结果如下。95百分位采用升序样本的最近秩定义。')
    tab([['统计指标','DPU调用耗时'],['最小值',f"{metrics['inference_ms']['min']:.3f} ms"],['平均值',f"{metrics['inference_ms']['mean']:.3f} ms"],['中位数',f"{metrics['inference_ms']['median']:.3f} ms"],['95百分位',f"{metrics['inference_ms']['p95_nearest_rank']:.3f} ms"],['最大值',f"{metrics['inference_ms']['max']:.3f} ms"]],[.5,.5])
    p('3.8.4 软件处理时延与持续更新能力','声界三级标题')
    p('实时运行测试使用MZ7100FA、XVF3800 USB音频输入及1280×720 HDMI输出，保留当前DPU＋HDMI版本的应用并行负载。环境声分析窗口为1秒，更新步长为0.5秒。本次计时只描述该平台的声音分类与显示运行链路，不计作Vosk语音转写延迟。')
    tab([['测量项目','本次结果'],['统计区间',f"{metrics['duration_s']:.3f}秒；{metrics['window_count']}个推理窗口"],['结果更新速率',f"{metrics['updates_per_second']:.3f}次/秒"],['相邻窗口时间间隔','中位数500 ms；范围499～501 ms'],['接收窗口后软件处理耗时','平均444.401 ms；中位数440 ms'],['软件处理耗时95百分位 / 最大值','465 ms / 744 ms'],['USB或DPU异常窗口','0 / 436'],['保留的稳态HDMI状态','102帧；USB、DPU、显示及数据新鲜度检查异常0帧'],['音频采集状态','记录1次至少570.714 ms的缓冲溢出，随后恢复'],['存储写入','本次读取时SD及eMMC写入计数均为0']],[.46,.54])
    p('软件处理耗时从最新音频窗口被应用接收开始，至分类和事件状态形成时结束；它不包含前面的1秒窗口积累，也不包含显示器扫描与液晶响应。因此该值用于评估板端处理负载，声源事件到画面出现的物理延迟需采用同步声光测量。DPU调用耗时与软件处理耗时的差别表明，当前更新周期主要由前后处理及应用调度决定。')
    p('3.8.5 片上功耗估算','声界三级标题')
    p('Vivado布局布线后的功耗报告给出的片上总功耗估算为5.136 W，其中动态功耗4.851 W、器件静态功耗0.285 W。该结果包含报告中的PS7与PL片上组件，采用工具活动率假设，报告置信度为Low；它是设计功耗估算，不是整机供电端实测功率，不包含麦克风、显示器等外部设备。')
    p('资源、时序及功耗数据来自同一已部署硬件版本的实现报告，运行统计来自2026年11月2日串口读取并保存的板端日志。两类结果共同描述DPU与显示硬件平台的实现规模和实际运行性能。')
    anchor=next(n for n in body if plain(n).startswith('第四部分'))
    for block in blocks:anchor.addprevious(block)
    parts['word/document.xml']=E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(OUTPUT,'w',ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    with ZipFile(OUTPUT) as z,ZipFile(SOURCE) as original:
        assert z.testzip() is None
        assert all(z.read(n)==original.read(n) for n in original.namelist() if n!='word/document.xml')
        final=E.fromstring(z.read('word/document.xml'))
        assert len(final.xpath('//m:oMath',namespaces=NS))==21
    assert all(E.tostring(n) in original_nodes for n in body if n not in blocks and n not in (summary,note))
    assert [E.tostring(n) for n in summary.findall('w:tr',NS)[:len(original_summary_rows)]]==original_summary_rows
    assert SOURCE.read_bytes()==source_bytes
    receipt={'source':str(SOURCE),'source_sha256':sha(source_bytes),'output':str(OUTPUT),'output_bytes':OUTPUT.stat().st_size,'output_sha256':sha(OUTPUT.read_bytes()),'static_sources':static_sources,'board_evidence':str(WORK/'current-board-evidence.json.gz'),'board_evidence_sha256':sha((WORK/'current-board-evidence.json.gz').read_bytes()),'metrics':metrics,'preserved_all_other_zip_parts':True,'preserved_existing_body_except_extended_summary_table':True,'native_math_count':21,'board_reprogrammed':False,'sd_modified':False,'scope':'User selected the DPU plus HDMI platform; angular measurement deferred by user; CER excluded.'}
    (WORK/'document-build.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(OUTPUT),'bytes':receipt['output_bytes'],'sha256':receipt['output_sha256']},ensure_ascii=True))


if __name__=='__main__':main()
