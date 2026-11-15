from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
from xml.sax.saxutils import escape
out=Path(r'D:\yolo26-tracking\output\figures')
labels=[['提取视频','首帧'],['识别静态','背景区域'],['排除车辆等','动态目标'],['标注多个','静态区域'],['合并生成','二值掩膜'],['缩放至运动','估计分辨率'],['应用于全部','视频帧']]
W,H=1580,290; scale=3
im=Image.new('RGB',(W*scale,H*scale),'white'); d=ImageDraw.Draw(im)
f=ImageFont.truetype(r'C:\Windows\Fonts\simsun.ttc',24*scale)
cf=ImageFont.truetype(r'C:\Windows\Fonts\simsun.ttc',25*scale)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',f'<rect width="{W}" height="{H}" fill="white"/>']
for i,lines in enumerate(labels):
 x=30+i*220; y=65; bw=200; bh=100
 box=tuple(v*scale for v in (x,y,x+bw,y+bh))
 radius=30 if i in (0,6) else 6
 d.rounded_rectangle(box,radius=radius*scale,outline='black',width=4)
 svg.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{radius}" fill="white" stroke="black" stroke-width="1.4"/>')
 for j,t in enumerate(lines):
  cy=y+34+j*34
  d.text(((x+bw/2)*scale,cy*scale),t,font=f,fill='black',anchor='mm')
  svg.append(f'<text x="{x+bw/2}" y="{cy}" dominant-baseline="central" text-anchor="middle" font-family="SimSun,宋体,serif" font-size="24">{escape(t)}</text>')
 if i<6:
  a=x+bw; b=x+220; cy=y+bh/2
  d.line((a*scale,cy*scale,(b-7)*scale,cy*scale),fill='black',width=4)
  d.polygon([(b*scale,cy*scale),((b-8)*scale,(cy-4)*scale),((b-8)*scale,(cy+4)*scale)],fill='black')
  svg.append(f'<path d="M{a} {cy} H{b-7}" stroke="black" stroke-width="1.4"/><path d="M{b} {cy} L{b-8} {cy-4} L{b-8} {cy+4} Z" fill="black"/>')
caption='静态背景掩膜构建流程图'
d.text((W*scale/2,235*scale),caption,font=cf,fill='black',anchor='mm')
svg.append(f'<text x="{W/2}" y="235" dominant-baseline="central" text-anchor="middle" font-family="SimSun,宋体,serif" font-size="25">{caption}</text></svg>')
im.save(out/'static_mask_workflow_horizontal.png',dpi=(300,300))
(out/'static_mask_workflow_horizontal.svg').write_text('\n'.join(svg),encoding='utf-8')
