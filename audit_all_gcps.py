# -*- coding: utf-8 -*-
"""Per-GCP robustness audit: residuals, leave-one-out, and random holdout."""
import json, csv, argparse
from pathlib import Path
import numpy as np
from coordinate_mapper import read_gcps

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gcps',required=True);ap.add_argument('--comparison',required=True);ap.add_argument('--random',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 g=read_gcps(a.gcps); comp=json.load(open(a.comparison,encoding='utf-8')); rnd=json.load(open(a.random,encoding='utf-8'))
 by={x['method']:x for x in comp}; rows=[]
 for i,r in enumerate(g):
  row={'id':r['id']}
  for n in ['affine','homography','polynomial_2','tps']:
   row[f'{n}_train_m']=by[n]['train_errors_m'][i]; row[f'{n}_loo_m']=by[n]['loo_errors_m'][i]
  # A transparent consensus score: number of non-TPS models with LOO error >=1m.
  vals=[row[f'{n}_loo_m'] for n in ['affine','homography','polynomial_2','tps']]
  row['loo_mean_all_m']=float(np.mean(vals)); row['loo_median_all_m']=float(np.median(vals)); row['models_loo_ge_1m']=sum(v>=1 for v in vals)
  for n in ['affine','homography']:
   best=rnd['best'][n]; row[f'{n}_selected_for_best']=r['id'] in best['train_ids']; row[f'{n}_best_holdout_m']=next((x for gid,x in zip(best['val_ids'],best['metrics']['val_errors_m']) if gid==r['id']),None)
  rows.append(row)
 # rank by consensus median, with broad agreement first
 rows.sort(key=lambda x:(-x['models_loo_ge_1m'],-x['loo_median_all_m']))
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out.with_suffix('.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
