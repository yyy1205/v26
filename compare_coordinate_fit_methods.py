# -*- coding: utf-8 -*-
"""Compare pixel-to-local-ENU fits using only the measured GCP CSV."""
from pathlib import Path
import csv, json
import cv2
import numpy as np
from coordinate_mapper import read_gcps, lonlat_to_enu

GCP_PATH = Path(r"F:\ybw\gps\gcps.csv")
OUT = Path("outputs/coordinate_mapper")


def design(p, degree):
    x, y = p[:, 0], p[:, 1]
    terms = [np.ones_like(x), x, y]
    if degree >= 2: terms += [x*y, x*x, y*y]
    if degree >= 3: terms += [x*x*y, x*y*y, x**3, y**3]
    return np.column_stack(terms)


def fit_poly(p, w, degree):
    # Normalize pixel coordinates for numerical stability, then solve two LS systems.
    center, scale = p.mean(0), p.std(0)
    q = (p - center) / scale
    A = design(q, degree)
    coef = np.linalg.lstsq(A, w, rcond=None)[0]
    return lambda z: design((np.asarray(z) - center) / scale, degree) @ coef


def fit_affine(p, w):
    M, _ = cv2.estimateAffine2D(p, w, method=cv2.LMEDS)
    return lambda z: cv2.transform(np.asarray(z, dtype=np.float64).reshape(-1, 1, 2), M).reshape(-1, 2)


def fit_homography(p, w):
    H, _ = cv2.findHomography(p, w, cv2.LMEDS)
    return lambda z: cv2.perspectiveTransform(np.asarray(z, dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def fit_tps(p, w, smoothing=0.0):
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:
        return None
    center, scale = p.mean(0), p.std(0)
    q = (p-center)/scale
    model = RBFInterpolator(q, w, kernel="thin_plate_spline", smoothing=smoothing)
    return lambda z: model((np.asarray(z)-center)/scale)


def score(name, fitter, p, w):
    pred = fitter(p)
    err = np.linalg.norm(pred-w, axis=1)
    loo = []
    for i in range(len(p)):
        try:
            f = fitter.__wrapped__(np.delete(p,i,0), np.delete(w,i,0)) if hasattr(fitter, '__wrapped__') else None
            if f is None: continue
            loo.append(float(np.linalg.norm(f(p[i:i+1])[0]-w[i])))
        except Exception:
            loo.append(float("nan"))
    return {"method": name, "train_mean_m": float(err.mean()), "train_rmse_m": float(np.sqrt(np.mean(err**2))),
            "train_max_m": float(err.max()), "train_errors_m": err.tolist(), "loo_errors_m": loo}


def factory(kind):
    def f(p,w):
        if kind == "affine": return fit_affine(p,w)
        if kind == "homography": return fit_homography(p,w)
        if kind.startswith("polynomial_"): return fit_poly(p,w,int(kind[-1]))
        return fit_tps(p,w,0.0)
    return f


def main():
    rows=read_gcps(GCP_PATH); lon0,lat0=rows[0]['longitude'],rows[0]['latitude']
    p=np.array([[r['pixel_x'],r['pixel_y']] for r in rows]); e,n=lonlat_to_enu([r['longitude'] for r in rows],[r['latitude'] for r in rows],lon0,lat0); w=np.c_[e,n]
    results=[]
    for kind in ["affine","homography","polynomial_2","polynomial_3","tps"]:
        fit=factory(kind); base=fit(p,w); pred=base(p); err=np.linalg.norm(pred-w,axis=1); loo=[]
        for i in range(len(p)):
            f=fit(np.delete(p,i,0),np.delete(w,i,0)); loo.append(float(np.linalg.norm(f(p[i:i+1])[0]-w[i])))
        results.append({"method":kind,"train_mean_m":float(err.mean()),"train_rmse_m":float(np.sqrt(np.mean(err**2))),"train_max_m":float(err.max()),"loo_mean_m":float(np.mean(loo)),"loo_rmse_m":float(np.sqrt(np.mean(np.array(loo)**2))),"loo_max_m":float(max(loo)),"train_errors_m":err.tolist(),"loo_errors_m":loo})
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'fit_methods_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if not isinstance(v,list)} for r in results],indent=2))

if __name__=='__main__': main()
