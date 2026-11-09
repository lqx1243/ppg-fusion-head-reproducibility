#!/usr/bin/env python3
"""Portable CPU rerun. Train all 450 trajectories before evaluating selected models.

This is a new run, not a continuation of the archived study. No cloud API is used.
"""
import argparse,copy,csv,json,math,platform,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import study as s
from data import Pool


def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def model(mode,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    return s.build_model(mode)


def predict(net,dataset,stats):
    net.eval();parts=[]
    with torch.no_grad():
        for x,_ in DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0):
            result=net(x).cpu().numpy()*float(stats['label_std'])+float(stats['label_mean'])
            parts.append(result.astype(np.float32))
    raw=np.concatenate(parts).astype(np.float32)
    if not np.isfinite(raw).all(): raise ValueError('Nonfinite prediction')
    return raw


def metrics(raw,y):
    clipped=np.clip(raw,30,220);e=clipped.astype(np.float64)-y.astype(np.float64);ae=np.abs(e)
    return dict(n=len(y),mae_bpm=float(ae.mean()),rmse_bpm=float(np.sqrt(np.square(e).mean())),bias_bpm=float(e.mean()),
        within5_pct=float(100*np.mean(ae<=5)),within10_pct=float(100*np.mean(ae<=10)),fail15_pct=float(100*np.mean(ae>15)),
        coverage_pct=100.,raw_out_of_range_pct=float(100*np.mean((raw<30)|(raw>220))),raw_out_of_range_n=int(np.sum((raw<30)|(raw>220))))


def train_one(pool,out,fold,roles,mode,seed,stats):
    target=out/'training'/fold/mode/f'seed_{seed}';target.mkdir(parents=True,exist_ok=False)
    net=model(mode,seed);optimizer=torch.optim.AdamW(net.parameters(),lr=s.HYPERPARAMS['lr'],weight_decay=s.HYPERPARAMS['weight_decay'])
    criterion=nn.SmoothL1Loss(beta=s.HYPERPARAMS['beta']);generator=torch.Generator().manual_seed(seed)
    train=pool.dataset(roles['train'],stats,s.model_spec(mode)['base']);inner=pool.dataset(roles['inner'],stats,s.model_spec(mode)['base'])
    labels=pool.metadata(pool.indices(roles['inner']))['y'];history=[];best=math.inf;choice=0;best_state=None;endpoints={}
    try:
        for epoch in range(1,41):
            order=torch.randperm(len(train),generator=generator).tolist()
            loader=DataLoader(train,batch_size=256,sampler=order,num_workers=0);losses=[];tick=time.monotonic();net.train()
            for x,y in loader:
                optimizer.zero_grad(set_to_none=True);loss=criterion(net(x),y)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss; failed run preserved.')
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            train_seconds=time.monotonic()-tick;tick=time.monotonic()
            score=s.selection_mae(predict(net,inner,stats),labels)
            row=dict(epoch=epoch,train_loss=float(np.mean(losses)),inner_mae_bpm=score,train_seconds=train_seconds,inner_seconds=time.monotonic()-tick)
            history.append(row);write(target/f'epoch_{epoch:03d}.json',row)
            if score<best-s.HYPERPARAMS['improvement_min']: best,choice,best_state=score,epoch,copy.deepcopy(net.state_dict())
            if epoch in s.ENDPOINTS:
                with (target/f'selected_{epoch}.pt').open('xb') as f: torch.save(best_state,f)
                endpoints[str(epoch)]=dict(selected_epoch=choice,inner_mae_bpm=best,checkpoint_sha256=s.sha(target/f'selected_{epoch}.pt'))
            print(f'{fold} {mode} {seed} {epoch}/40: inner MAE {score:.6f}',flush=True)
    finally: train.close();inner.close()
    selected=s.select_checkpoints(history)
    for ep in s.ENDPOINTS:
        assert selected[ep]['selected_epoch']==endpoints[str(ep)]['selected_epoch']
    record=dict(fold=fold,mode=mode,seed=seed,history=history,endpoints=endpoints,status='complete')
    write(target/'record.json',record)
    return record


def verify_cache(cache):
    expected=json.loads((ROOT/'provenance/cache_sha256.json').read_text())
    actual={k:s.sha(cache/k) for k in expected}
    if any(actual[k]!=v['sha256'] for k,v in expected.items()): raise ValueError('Input cache SHA-256 differs; investigate before training.')
    return actual


def init(a):
    if platform.system()!='Linux': raise ValueError('Formal rerun requires Linux CPU; use synthetic tests on other hosts.')
    if torch.version.cuda is not None: raise ValueError('Install CPU-only PyTorch.')
    if a.run.exists(): raise ValueError('Choose a new run directory; no automatic continuation/retry.')
    hashes=verify_cache(a.cache);a.run.mkdir(parents=True);torch.set_num_threads(8);torch.set_num_interop_threads(1)
    with Pool(a.cache) as pool:
        stats={fold:pool.fit_stats(pool.indices(roles['train'])) for fold,roles in s.FOLDS.items()}
        write(a.run/'protocol.json',dict(configuration=s.CONFIG,folds=s.FOLDS,normalization=stats,cache_sha256=hashes,
            python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,intra_threads=8,inter_threads=1,
            sources={p.relative_to(ROOT).as_posix():s.sha(p) for d in ('src','scripts','configs','splits') for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}))
        records=[]
        for fold,roles in s.FOLDS.items():
            for seed in s.SEEDS:
                for mode in s.MODES: records.append(train_one(pool,a.run,fold,roles,mode,seed,stats[fold]))
        assert len(records)==450
        # Verify every selected file and replay every selector before opening outer model inputs.
        manifest={}
        for record in records:
            base=a.run/'training'/record['fold']/record['mode']/f"seed_{record['seed']}"
            replay=s.select_checkpoints(record['history'])
            for ep in s.ENDPOINTS:
                assert replay[ep]['selected_epoch']==record['endpoints'][str(ep)]['selected_epoch']
                assert s.sha(base/f'selected_{ep}.pt')==record['endpoints'][str(ep)]['checkpoint_sha256']
            manifest[str(base.relative_to(a.run))]=s.sha(base/'record.json')
        write(a.run/'training_frozen.json',manifest)
        if verify_cache(a.cache)!=hashes: raise ValueError('Inputs changed during training')
        results=[]
        for record in records:
            fold,mode,seed=record['fold'],record['mode'],record['seed'];roles=s.FOLDS[fold];base=a.run/'training'/fold/mode/f'seed_{seed}'
            for ep in s.ENDPOINTS:
                net=model(mode,seed);net.load_state_dict(torch.load(base/f'selected_{ep}.pt',map_location='cpu',weights_only=True))
                endpoint=a.run/'evaluation'/fold/mode/f'seed_{seed}'/f'endpoint_{ep}';endpoint.mkdir(parents=True)
                for role in ('train','inner','outer'):
                    data=pool.dataset(roles[role],stats[fold],s.model_spec(mode)['base'])
                    try: pred=predict(net,data,stats[fold])
                    finally: data.close()
                    meta=pool.metadata(pool.indices(roles[role]));np.savez_compressed(endpoint/(role+'.npz'),raw=pred,**meta)
                    if role=='outer': results.append(dict(endpoint=ep,mode=mode,seed=seed,subject=roles[role][0],**metrics(pred,meta['y'])))
        with (a.run/'participant_by_seed.csv').open('x') as f:
            w=csv.DictWriter(f,results[0].keys(),lineterminator='\n');w.writeheader();w.writerows(results)
        write(a.run/'completed.json',dict(trajectories=450,endpoints=900,prediction_vectors=2700,training_frozen_sha256=s.sha(a.run/'training_frozen.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();init(a)
if __name__=='__main__': main()
