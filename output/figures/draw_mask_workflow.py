from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import html
out=Path(r'D:\yolo26-tracking\output\figures')
labels=['提取视频首帧','识别静态背景区域','排除车辆和其他动态目标','标注多个静态区域','合并为二值掩膜','缩放至运动估计分辨率','应用于所有视频帧']
w,h=720,1030
im=Image.new('RGB',(w*3,h*3),'white'); d=ImageDraw.Draw(im)
font=ImageFont.truetype(r'C:\Windows\Fonts\simsun.ttc',26*3)
captionfont=ImageFont.truetype(r'C:\Windows\Fonts\simsun.ttc',25*3)
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="1030" viewBox="0 0 720 1030">','<rect width="720" height="1030" fill="white"/>','<g fill="white" stroke="black" stroke-width="1.6">']
for i,t in enumerate(labels):
 y=45+i*130; x=135; bw=450; bh=64
 box=(x*3,y*3,(x+bw)*3,(y+bh)*3)
 if i in [0,6]:
  d.rounded_rectangle(box,radius=32*3,outline='black',width=5)
  svg.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="32"/>')
 else:
  d.rectangle(box,outline='black',width=5)
  svg.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}"/>')
 d.text((w*1.5,(y+32)*3),t,font=font,fill='black',anchor='mm')
 if i<6:
  ya=y+bh; ye=y+130
  d.line((360*3,ya*3,360*3,(ye-12)*3),fill='black',width=5)
  d.polygon([(354*3,(ye-13)*3),(366*3,(ye-13)*3),(360*3,ye*3)],fill='black')
  svg.append(f'<path d="M360 {ya} V{ye-12}" fill="none"/><path d="M354 {ye-13} L366 {ye-13} L360 {ye} Z" fill="black" stroke="none"/>')
svg.append('</g><g font-family="SimSun,宋体,serif" font-size="26" fill="black" text-anchor="middle">')
for i,t in enumerate(labels): svg.append(f'<text x="360" y="{45+i*130+41}">{html.escape(t)}</text>')
caption='静态背景掩膜构建流程图'
d.text((1080,960*3),caption,font=captionfont,fill='black',anchor='mm')
svg.append(f'<text x="360" y="969" font-size="25">{caption}</text></g></svg>')
(out/'static_mask_workflow.svg').write_text('\n'.join(svg),encoding='utf-8')
im.save(out/'static_mask_workflow.png',dpi=(300,300))
im.save(out/'static_mask_workflow.pdf',resolution=300)
print(out)
