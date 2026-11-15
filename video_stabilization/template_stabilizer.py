import cv2,numpy as np
class StaticTemplate:
    def __init__(self,nfeatures=1200,ratio=.7): self.sift=cv2.SIFT_create(nfeatures=nfeatures); self.ratio=ratio; self.ref=None; self.k=None; self.d=None
    def build(self,frames):
        self.ref=frames[0]; self.k,self.d=self.sift.detectAndCompute(self.ref,None); return len(self.k)
    def locate(self,frame,model='affine'):
        k,d=self.sift.detectAndCompute(frame,None); info={'keypoints':len(k),'matches':0,'inliers':0,'inlier_ratio':0.,'valid':False}
        if self.d is None or d is None:return np.eye(3,dtype=np.float32),info
        ms=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=40)).knnMatch(self.d,d,k=2); good=[m for m,n in ms if m.distance<self.ratio*n.distance]; info['matches']=len(good)
        if len(good)<8:return np.eye(3,dtype=np.float32),info
        p=np.float32([self.k[m.queryIdx].pt for m in good]); q=np.float32([k[m.trainIdx].pt for m in good]);
        if model=='homography': M,inl=cv2.findHomography(p,q,cv2.RANSAC,3.)
        else:
            A,inl=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3.); M=np.vstack([A,[0,0,1]]) if A is not None else None
        if M is None:return np.eye(3,dtype=np.float32),info
        info.update(inliers=int(inl.sum()),inlier_ratio=float(inl.mean()),valid=True); return M.astype(np.float32),info

class MultiReference:
    def __init__(self, interval=50, max_refs=5): self.interval=interval; self.max_refs=max_refs; self.refs=[]
    def add(self, frame, index, global_pose=None):
        t=StaticTemplate(); t.build(frame); pose=np.eye(3,dtype=np.float32) if global_pose is None else np.asarray(global_pose,np.float32); self.refs.append((index,t,pose)); self.refs=self.refs[-self.max_refs:]
    def locate(self, frame):
        results=[]
        for idx,t,gref in self.refs:
            M,info=t.locate(frame,'affine')
            if info['valid'] and info['inlier_ratio']>=.5: results.append((M@gref,info,idx))
        if not results: return np.eye(3,dtype=np.float32),{'valid':False,'references_used':0}
        params=[]
        for M,_,_ in results:
            params.append([M[0,2],M[1,2],np.arctan2(M[1,0],M[0,0]),np.hypot(M[0,0],M[1,0])])
        tx,ty,a,s=np.median(np.asarray(params),axis=0); M=np.array([[s*np.cos(a),-s*np.sin(a),tx],[s*np.sin(a),s*np.cos(a),ty],[0,0,1]],np.float32); return M,{'valid':True,'references_used':len(results),'inlier_ratio':float(np.median([x[1]['inlier_ratio'] for x in results]))}

def visualize_matches(reference,current,path,ratio=.7):
    sift=cv2.SIFT_create(nfeatures=1200); k1,d1=sift.detectAndCompute(reference,None); k2,d2=sift.detectAndCompute(current,None); ms=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=40)).knnMatch(d1,d2,k=2); good=[m for m,n in ms if m.distance<ratio*n.distance]
    p=np.float32([k1[m.queryIdx].pt for m in good]); q=np.float32([k2[m.trainIdx].pt for m in good]); _,inl=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3.); mask=inl.ravel().tolist() if inl is not None else None
    vis=cv2.drawMatches(reference,k1,current,k2,good,None,matchColor=(0,255,0),singlePointColor=(0,0,255),matchesMask=mask,flags=2); cv2.imwrite(str(path),vis); return {'matches':len(good),'inliers':sum(mask) if mask else 0}

def pose_matrix(tx,ty,angle_deg,scale=1.0):
    a=np.deg2rad(angle_deg); return np.array([[scale*np.cos(a),-scale*np.sin(a),tx],[scale*np.sin(a),scale*np.cos(a),ty],[0,0,1]],np.float64)

def conservative_corrections(poses,window=125,translation_deadband=.5,rotation_deadband=.02,max_translation_step=.1,max_rotation_step=.001):
    """Return target@inv(observed) matrices without modifying high-frequency motion."""
    p=np.asarray(poses,float); n=len(p); w=min(max(3,int(window)),n); k=np.ones(w)/w
    low=np.column_stack([np.convolve(np.pad(p[:,j],(w//2,w-1-w//2),mode='edge'),k,'valid') for j in range(4)])
    target=p.copy(); drift=low-p; drift[np.abs(drift[:,0])<translation_deadband,0]=0; drift[np.abs(drift[:,1])<translation_deadband,1]=0; drift[np.abs(drift[:,2])<rotation_deadband,2]=0
    drift[:,0]=np.clip(drift[:,0],-max_translation_step,max_translation_step); drift[:,1]=np.clip(drift[:,1],-max_translation_step,max_translation_step); drift[:,2]=np.clip(drift[:,2],-max_rotation_step,max_rotation_step); target[:,:3]+=drift[:,:3]
    return [pose_matrix(*target[i])@np.linalg.inv(pose_matrix(*p[i])) for i in range(n)]

def center_conjugate(M,width,height):
    """Apply a correction around image center rather than the top-left origin."""
    C=np.array([[1,0,width/2],[0,1,height/2],[0,0,1]],float); return C@M@np.linalg.inv(C)
