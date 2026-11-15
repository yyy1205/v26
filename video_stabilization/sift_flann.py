import cv2,numpy as np
def estimate(prev,curr,ratio=.7,min_matches=8):
    sift=cv2.SIFT_create(nfeatures=1500); k1,d1=sift.detectAndCompute(prev,None); k2,d2=sift.detectAndCompute(curr,None)
    info={'keypoints':len(k2),'matches':0,'inliers':0,'ratio':0.0,'valid':False}
    if d1 is None or d2 is None:return np.eye(2,3,dtype=np.float32),info
    flann=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=50)); ms=flann.knnMatch(d1,d2,k=2); good=[m for m,n in ms if m.distance<ratio*n.distance]; info['matches']=len(good)
    if len(good)<min_matches:return np.eye(2,3,dtype=np.float32),info
    p=np.float32([k1[m.queryIdx].pt for m in good]); q=np.float32([k2[m.trainIdx].pt for m in good]); M,inl=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.0)
    if M is None:return np.eye(2,3,dtype=np.float32),info
    info['inliers']=int(inl.sum()); info['ratio']=info['inliers']/len(good); info['valid']=True; return M.astype(np.float32),info
