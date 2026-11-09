#!/usr/bin/env python3
"""Prepare the exact four-channel cache from official PPG-DaLiA participant PKLs.

Only open PKL files obtained from the trusted official dataset: pickle can execute
code. No download, training, participant selection or model inference occurs.
"""
import argparse,csv,json,pickle,sys
from pathlib import Path
import numpy as np
from scipy import signal
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from study import sha


def aligned_wrist(raw):
    # Preserve the original two float32 storage boundaries.
    ppg=np.asarray(raw['signal']['wrist']['BVP'],dtype=np.float64).reshape(-1).astype(np.float32)
    native=np.asarray(raw['signal']['wrist']['ACC'],dtype=np.float64).astype(np.float32)
    acc=signal.resample_poly(native.astype(np.float64),2,1,axis=0).astype(np.float32)
    labels=np.asarray(raw['label']).reshape(-1).astype(np.float32)
    activities=np.repeat(np.asarray(raw['activity']).reshape(-1).astype(np.int8),16)
    return ppg,acc,labels,activities


def preprocess(ppg,acc,starts):
    index=starts[:,None]+np.arange(512)
    pulse=np.asarray(ppg[index],dtype=np.float64)
    sos=signal.butter(4,[.5,4],btype='band',fs=64,output='sos')
    pulse=signal.sosfiltfilt(sos,pulse,axis=1)
    pulse-=pulse.mean(axis=1,keepdims=True)
    motion=np.asarray(acc[index],dtype=np.float64)
    motion-=motion.mean(axis=1,keepdims=True)
    return np.concatenate((pulse[:,None,:],motion.transpose(0,2,1)),axis=1).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True,help='Directory containing S1/S1.pkl through S15/S15.pkl')
    p.add_argument('--output',type=Path,required=True,help='New preparation directory; includes cache and activity CSV')
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists; use a new directory to preserve previous work.')
    parts=json.loads((ROOT/'splits/storage_partitions.json').read_text())
    hashes=json.loads((ROOT/'provenance/cache_sha256.json').read_text())
    a.output.mkdir(parents=True);cache=a.output/'cache';cache.mkdir();activity=[];origins={}
    # Historical partition names only preserve storage ordering; LOSO roles are separate.
    for part,people in parts.items():
        folder=cache/part;folder.mkdir()
        count=38964 if part=='train' else 12879 if part=='val' else 12854
        arr={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=dt,
             shape=(count,4,512) if k=='x' else (count,)) for k,dt in [('x','float32'),('y','float32'),('subject','int16'),('window','int32')]}
        pos=0
        for name in people:
            source=a.raw/name/(name+'.pkl')
            if not source.is_file(): raise FileNotFoundError(source)
            origins[source.relative_to(a.raw).as_posix()]=sha(source)
            with source.open('rb') as f: raw=pickle.load(f,encoding='latin1')
            if str(raw['subject'])!=name: raise ValueError('Participant identity mismatch')
            ppg,acc,labels,act=aligned_wrist(raw);person=int(name[1:]);windows=np.arange(len(labels),dtype=np.int32)
            if np.any(windows.astype(np.int64)*128+512>len(ppg)) or len(acc)<len(ppg):
                raise ValueError('Incomplete supplied label windows; do not silently drop them')
            for start in range(0,len(labels),256):
                stop=min(start+256,len(labels));dest=slice(pos+start,pos+stop)
                arr['x'][dest]=preprocess(ppg,acc,windows[start:stop].astype(np.int64)*128)
                arr['y'][dest]=labels[start:stop];arr['subject'][dest]=person;arr['window'][dest]=windows[start:stop]
            for w in windows:
                seg=act[int(w)*128:int(w)*128+512]
                activity.append(dict(subject=person,window=int(w),activity_code=int(np.bincount(seg).argmax()),reference_bpm=float(labels[w])))
            pos+=len(labels);print(f'{name}: {len(labels)} windows',flush=True)
        if pos!=count: raise ValueError('Unexpected window count')
        for v in arr.values(): v.flush()
        del arr
    checks={k:dict(expected_sha256=v['sha256'],actual_sha256=sha(cache/k)) for k,v in hashes.items()}
    report=dict(raw_sha256=origins,cache_checks=checks,all_match=all(x['expected_sha256']==x['actual_sha256'] for x in checks.values()),
                numpy=np.__version__,scipy=__import__('scipy').__version__)
    (a.output/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'activities.csv').open('w') as f:
        w=csv.DictWriter(f,activity[0].keys(),lineterminator='\n');w.writeheader();w.writerows(sorted(activity,key=lambda r:(r['subject'],r['window'])))
    if not report['all_match']: raise ValueError('Cache differs from historical inputs; outputs retained for investigation.')
    print('All 12 cache files match the historical SHA-256 manifest.')

if __name__=='__main__': main()
