"""Conservative nadir drift estimator: phase correlation + optional ECC."""
import cv2,numpy as np
def estimate_pair(prev,curr,mask=None,mode='translation'):
    p=prev.astype(np.float32); c=curr.astype(np.float32); mask=None if mask is None else mask.astype(np.float32)/255.0; shift,response=cv2.phaseCorrelate(p,c,mask)
    if mode=='translation': return np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32),float(response)
    mm=cv2.MOTION_EUCLIDEAN if mode=='euclidean' else cv2.MOTION_AFFINE; warp=np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32)
    try: cc,_=cv2.findTransformECC(p,c,warp,mm,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,50,1e-5),mask,5); return cc,float(_)
    except cv2.error: return warp,float(response)
def low_frequency_correction(values,window=125,deadband=(.3,.03,.002)):
    v=np.asarray(values,float); out=v.copy()
    for j in range(v.shape[1]):
        k=np.ones(max(3,int(window)))/max(3,int(window)); lp=np.convolve(np.pad(v[:,j],(len(k)//2,)*2,mode='edge'),k,'valid'); d=lp-v[:,j]; out[:,j]=np.where(np.abs(d)<deadband[j],0,d)
    return out
