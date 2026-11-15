import json,subprocess,sys,cv2
from pathlib import Path
FFPROBE=r'C:\Tools\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffprobe.exe'
def verify(path,out_json):
    r=subprocess.run([FFPROBE,'-v','error','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True)
    Path(out_json).write_text(r.stdout,encoding='utf-8')
    if r.returncode: raise RuntimeError(r.stderr)
    c=cv2.VideoCapture(str(path)); frames=0; bad=0
    while True:
        ok,f=c.read()
        if not ok: break
        frames+=1; bad+=int(f is None or f.size==0)
    meta=json.loads(r.stdout); c.release(); return {'decoded_frames':frames,'empty_frames':bad,'streams':meta.get('streams',[]),'format':meta.get('format',{})}
if __name__=='__main__': print(verify(sys.argv[1],sys.argv[2]))
