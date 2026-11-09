#!/usr/bin/env python3
"""Regenerate manuscript tables, figures and diagnostic summaries without training."""
import argparse,csv,json,sys,subprocess,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from recompute_tables import calculate,write_csv,load_csv


def verify_inputs():
    expected=json.loads((ROOT/'provenance/input_sha256.json').read_text())
    for name,digest in expected.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError('Input changed: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tables-only',action='store_true');p.add_argument('--conference',action='store_true',help='Also regenerate the compact ICAID figures');a=p.parse_args()
    if a.tables_only and a.conference: p.error('--tables-only and --conference cannot be combined')
    verify_inputs();out=ROOT/'outputs';out.mkdir(exist_ok=True)
    tables=calculate(ROOT/'inputs');dest=out/'tables';dest.mkdir(exist_ok=True)
    for name,rows in tables.items(): write_csv(dest/(name+'.csv'),rows)
    core=out/'summaries/core';activity=out/'summaries/activity';core.mkdir(parents=True,exist_ok=True);activity.mkdir(parents=True,exist_ok=True)
    write_csv(core/'overall_summary.csv',tables['S1_full_metrics']);write_csv(core/'model_complexity.csv',tables['S6_model_cost'])
    write_csv(activity/'activity_summary.csv',tables['S5_activity_summary'])
    people={(int(r['endpoint']),r['mode'],int(r['seed']),int(r['subject'])):float(r['mae_bpm']) for r in load_csv(ROOT/'inputs/participant_by_seed.csv')}
    contrasts={'fusion_linear':[(1,'r2_linear'),(-1,'r1_linear')],'primary_fusion_mlp':[(1,'r2_mlp'),(-1,'r1_mlp')],
      'secondary_head_by_fusion':[(1,'r2_mlp'),(-1,'r1_mlp'),(-1,'r2_linear'),(1,'r1_linear')],
      'secondary_acc_r2_mlp':[(1,'r2_mlp'),(-1,'r0_mlp')]}
    participant=[];cohort=[]
    for ep in (12,40):
        for name,terms in contrasts.items():
            for person in range(1,16): participant.append(dict(endpoint=ep,contrast=name,person=person,**{f'seed_{seed}_mae_bpm':sum(sign*people[ep,mode,seed,person] for sign,mode in terms) for seed in range(3001,3006)}))
            for seed in range(3001,3006): cohort.append(dict(endpoint=ep,contrast=name,seed=seed,weighting='participant_equal',mae_bpm=sum(sum(sign*people[ep,mode,seed,person] for sign,mode in terms) for person in range(1,16))/15))
    write_csv(core/'participant_contrasts.csv',participant);write_csv(core/'contrasts_by_seed.csv',cohort)
    import format_tables
    overall,activities,costs,counts=format_tables.load_results()
    for name,pair in [('overall_table',format_tables.overall_fragments(overall,costs)),('activity_table',format_tables.activity_fragments(activities,counts))]:
        for suffix,content in zip(('tex','md'),pair): (dest/(name+'.'+suffix)).write_text(content)
    if not a.tables_only:
        for script in ('plot_signals.py','plot_architecture.py','plot_results.py','plot_cases.py'):
            subprocess.run([sys.executable,str(ROOT/'src'/script)],check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/training_diagnostics.py'),'--history-csv',str(ROOT/'inputs/training_history.csv'),'--selected-csv',str(ROOT/'inputs/training_selected_epochs.csv'),'--output-dir',str(out/'training'),'--figure-dir',str(out/'figures')],check=True)
    if a.conference:
        conference=out/'conference';conference.mkdir(exist_ok=True)
        commands=[
            ('conference_plot_signals.py',['--input',str(ROOT/'inputs/signal_example.npz'),'--output',str(conference)]),
            ('conference_plot_architecture.py',['--input',str(ROOT/'inputs/signal_example.npz'),'--output',str(conference)]),
            ('conference_plot_results.py',['--inputs',str(core),'--output',str(conference)]),
            ('conference_plot_cases.py',['--inputs',str(ROOT/'inputs'),'--output',str(conference)]),
            ('conference_plot_training_curve.py',['--input',str(out/'training/training_validation_curve_summary.csv'),'--output',str(conference/'training_validation_curves.pdf')]),
        ]
        for script,arguments in commands:
            subprocess.run([sys.executable,str(ROOT/'src'/script),*arguments],check=True)
    print('Reproduction complete:',out)
if __name__=='__main__': main()
