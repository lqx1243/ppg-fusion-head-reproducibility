#!/usr/bin/env python3
"""Export a completed portable rerun for tables and validation diagnostics.

The archived inputs are never overwritten. Figure 1/2 illustration and Figure 5
post-hoc windows remain archived examples unless a separate new selection is made.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from recompute_tables import write_csv,load_csv
from train import metrics
ACTIVITIES=('transient','sitting','stairs','table_soccer','cycling','driving','lunch','walking','working')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--activities',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Choose a new output directory.')
    done=json.loads((a.run/'completed.json').read_text());assert done['trajectories']==450 and done['endpoints']==900
    activity={(int(r['subject']),int(r['window'])):int(r['activity_code']) for r in load_csv(a.activities)}
    histories=[];selected=[];cells=[]
    for rec in sorted((a.run/'training').glob('outer_S*/*/seed_*/record.json')):
        r=json.loads(rec.read_text());person=int(r['fold'].split('S')[1]);identity=dict(mode=r['mode'],seed=r['seed'],person=person)
        for row in r['history']: histories.append(dict(**identity,**row))
        for ep in (12,40):
            sel=r['endpoints'][str(ep)];last=r['history'][ep-1]
            selected.append(dict(endpoint=ep,**identity,selected_epoch=sel['selected_epoch'],selected_inner_float32_mae_bpm=sel['inner_mae_bpm'],prefix_last_inner_float32_mae_bpm=last['inner_mae_bpm'],prefix_last_train_loss=last['train_loss']))
            path=a.run/'evaluation'/r['fold']/r['mode']/f"seed_{r['seed']}"/f'endpoint_{ep}'/'outer.npz'
            with np.load(path) as z:
                assert np.all(z['subject']==person);codes=np.array([activity[person,int(w)] for w in z['window']])
                for code,name in enumerate(ACTIVITIES):
                    mask=codes==code
                    m=metrics(z['raw'][mask],z['y'][mask]) if mask.any() else dict(n=0,mae_bpm='',rmse_bpm='',bias_bpm='',within5_pct='',within10_pct='',fail15_pct='',coverage_pct='',raw_out_of_range_pct='',raw_out_of_range_n=0)
                    cells.append(dict(endpoint=ep,subject=person,activity_code=code,activity_name=name,mode=r['mode'],seed=r['seed'],**m))
    assert len(histories)==18000 and len(selected)==900 and len(cells)==8100
    a.output.mkdir(parents=True)
    write_csv(a.output/'training_history.csv',histories);write_csv(a.output/'training_selected_epochs.csv',selected);write_csv(a.output/'person_activity_metrics.csv',cells)
    for name in ('architecture.json',): (a.output/name).write_bytes((ROOT/'inputs'/name).read_bytes())
    (a.output/'participant_by_seed.csv').write_bytes((a.run/'participant_by_seed.csv').read_bytes())
    print(a.output)
if __name__=='__main__':main()
