import cv2, numpy as np
from .stabilizer import StabilizationConfig, estimate_motion, stabilize_frames

def test_empty_descriptors_and_short_match_fallback():
    cfg=StabilizationConfig(); blank=np.zeros((100,100),np.uint8)
    M,info=estimate_motion(blank,blank,cfg)
    assert M.shape==(2,3) and not info['valid']

def test_smoothing_short_sequence():
    frames=[np.full((80,120,3),i*3,np.uint8) for i in range(3)]
    out,logs=stabilize_frames(frames,StabilizationConfig(smooth_window=45))
    assert len(out)==len(frames) and len(logs)==3

def test_video_io(tmp_path):
    path=str(tmp_path/'in.mp4'); outp=str(tmp_path/'out.mp4')
    w=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),7,(64,48))
    for i in range(5): w.write(np.full((48,64,3),i*20,np.uint8))
    w.release(); from .stabilizer import stabilize_video
    r=stabilize_video(path,outp,tmp_path/'log.json',StabilizationConfig(estimate_width=64))
    assert r['frames']==5 and abs(r['fps']-7)<.1
