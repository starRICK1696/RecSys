"""Generate PDF-only report figures from recorded KT-2 experiment outputs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL = {'unit':'Unit weights','trim':'Trim large errors','gaussian':'Gaussian','student':'Student-t (df=3)','cauchy':'Cauchy (df=1)'}


def read(path):
    if not path.exists(): return []
    with path.open() as f: rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in row.items():
            try: row[key]=float(value)
            except (ValueError,TypeError): pass
    return rows


def plot(directory):
    metrics_path = directory / 'metrics.csv'
    data = read(metrics_path)
    if not data:
        raise ValueError(f'No completed evaluations in {metrics_path}')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    target=directory/'figures'; target.mkdir(exist_ok=True)
    seeds=sorted({r['seed'] for r in data})
    caption=f'MovieLens-1M {"subset" if manifest["subset"] else "eligible cohort"}: {manifest["shape"][0]} users, {manifest["shape"][1]} items; {len(seeds)} seeds'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
    def save(fig,name,title):
        fig.suptitle(title+'\n'+caption,fontsize=13)
        fig.tight_layout(rect=(0,.04,1,.9))
        note = ('Boxes: quartiles and median; whiskers: 1.5 IQR; events pooled across seeds.'
                if name == '09_weight_distributions' else
                'Exploratory results. Bands: observed min–max across seeds, not confidence intervals.')
        fig.text(.5,.015,note,ha='center',fontsize=8)
        fig.savefig(target/f'{name}.pdf'); plt.close(fig)
    def curves(ax,rows,x,y):
        models=list(dict.fromkeys(r['model'] for r in rows))
        for name in models:
            selected=[r for r in rows if r['model']==name and isinstance(r.get(y),float)]
            xs=sorted({r[x] for r in selected})
            if not xs: continue
            groups=[np.array([r[y] for r in selected if r[x]==v]) for v in xs]
            means=[v.mean() for v in groups]
            line,=ax.plot(xs,means,'o-',markersize=4,label=LABEL.get(name,name))
            ax.fill_between(xs,[v.min() for v in groups],[v.max() for v in groups],color=line.get_color(),alpha=.1)
        ax.grid(alpha=.2); ax.set_ylabel(y); ax.set_xlabel(x)
    robust=[r for r in data if r['suite']=='robustness']
    if robust:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,nt in zip(axes,['noisy_1','noisy_0']):
            curves(ax,[r for r in robust if r['noise_type']==nt],'noise','ndcg');ax.set_title(nt);ax.set_ylabel('NDCG@20');ax.set_xlabel('Noise / original training edges')
        axes[1].legend(fontsize=8);save(fig,'01_robustness','Ranking robustness')
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        det=[r for r in robust if r['noise_type']=='noisy_1' and r['noise']>0 and r['model'] in LABEL]
        for ax,metric in zip(axes,['auroc','ap']):
            curves(ax,det,'noise',metric);ax.set_ylabel(metric.upper())
            if metric=='auroc': ax.axhline(.5,ls='--',color='gray',label='Random AUROC')
            else:
                xs=sorted({r['noise'] for r in det});ax.plot(xs,[np.mean([r['prevalence'] for r in det if r['noise']==x]) for x in xs],'k--',label='Added-event prevalence')
        axes[1].legend(fontsize=8);save(fig,'02_detection','Detection of injected events (not all real noise)')
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,nt in zip(axes,['noisy_1','noisy_0']):
            curves(ax,[r for r in robust if r['noise_type']==nt],'noise','train_seconds');ax.set_title(nt);ax.set_ylabel('Training time, seconds')
        axes[1].legend(fontsize=8);save(fig,'03_runtime','Training cost')
        delta=[]
        for r in robust:
            baseline=next(v for v in robust if v['model']=='unit' and v['seed']==r['seed'] and v['noise_type']==r['noise_type'] and v['noise']==r['noise'])
            delta.append({**r,'delta':r['ndcg']-baseline['ndcg']})
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,nt in zip(axes,['noisy_1','noisy_0']):
            curves(ax,[r for r in delta if r['noise_type']==nt and r['model']!='unit'],'noise','delta');ax.axhline(0,color='gray',ls='--');ax.set_title(nt);ax.set_ylabel('Paired delta NDCG@20 vs unit weights')
        axes[1].legend(fontsize=8);save(fig,'04_paired_delta','Paired ranking differences at the same seed')
        with (directory/'paired_deltas.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['model','seed','noise_type','noise','delta']);writer.writeheader();writer.writerows([{k:r[k] for k in writer.fieldnames} for r in delta])
    hybrid=[r for r in data if r['suite']=='hybrid']
    if hybrid:
        fig,ax=plt.subplots(figsize=(10,5));curves(ax,hybrid,'history','ndcg');ax.set_xticks([5,10,20]);ax.set_ylabel('NDCG@20');ax.set_xlabel('Available history');ax.legend(fontsize=8)
        save(fig,'05_hybrid','Genres × weights and capacity controls; no injected noise')
    stability=[r for r in data if r['suite']=='stability']
    if stability:
        fig,axes=plt.subplots(1,3,figsize=(14,5))
        for ax,method in zip(axes,['gaussian','student','cauchy']):
            subset=[{**r,'model':f'{r["initialization"]} {int(r["mixture_seed"])}'} for r in stability if r['model']==method]
            curves(ax,subset,'outer','ndcg');ax.set_title(LABEL[method]);ax.set_xlabel('Outer iterations');ax.set_ylabel('NDCG@20')
        axes[-1].legend(fontsize=8);save(fig,'06_stability','Mixture initialization and outer-loop sensitivity; 10% Noisy 1')
    cross=[r for r in data if r['suite']=='crossfit']
    if cross:
        fig,axes=plt.subplots(1,3,figsize=(13,5))
        for ax,metric in zip(axes,['ndcg','auroc','ap']):
            curves(ax,cross,'folds',metric);ax.set_xticks([0,3],['Training errors','3-fold errors']);ax.set_xlabel('Error source')
        axes[-1].legend(fontsize=8);save(fig,'07_crossfit','Errors on training events versus excluded events; 10% Noisy 1')
    groups=read(directory/'groups.csv')
    if groups:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,axis in zip(axes,['popularity','history']):
            curves(ax,[r for r in groups if r['axis']==axis],'group','ndcg');ax.set_xticks([0,1,2],['Low','Middle','High']);ax.set_xlabel('Target popularity' if axis=='popularity' else 'User history length');ax.set_ylabel('NDCG@20')
        axes[-1].legend(fontsize=8);save(fig,'08_groups','Ranking by training-derived groups; 10% Noisy 1')
    files=list(directory.glob('weights_*.npz'))
    if files:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for col,method in enumerate(['gaussian','student','cauchy']):
            arrays=[np.load(f) for f in files if f.name.startswith(f'weights_{method}_')]
            if not arrays: continue
            weights=np.concatenate([a['weights'] for a in arrays]);added=np.concatenate([a['added'] for a in arrays])
            for row,axis in enumerate(['popularity','history']):
                if not all(axis+'_group' in a for a in arrays):
                    raise ValueError('Weight files use an older protocol. Rerun experiments into pilot_v2.')
                membership=np.concatenate([a[axis+'_group'] for a in arrays])
                ax=axes[row,col]
                for flag,color,offset,label in [(False,'#3465a4',-.17,'Original'),(True,'#c65d26',.17,'Injected')]:
                    series=[weights[(membership==g)&(added==flag)] for g in range(3)]
                    bp=ax.boxplot(series,positions=np.arange(3)+offset,widths=.28,showfliers=False,patch_artist=True)
                    for box in bp['boxes']:box.set(facecolor=color,alpha=.6)
                    ax.plot([],[],color=color,label=label)
                ax.set_xticks([0,1,2],['Low','Middle','High']);ax.set_ylim(-.05,1.05);ax.set_title(LABEL[method]);ax.set_xlabel(axis);ax.set_ylabel('Weight')
        axes[0,-1].legend(fontsize=8)
        save(fig,'09_weight_distributions','Event-weight distributions pooled across seeds; 10% Noisy 1')
    # Tables are means and sample standard deviations, not significance tests.
    summary=[]
    keys=['suite','model','noise_type','noise','history','outer','initialization','mixture_seed','folds']
    buckets={}
    for r in data:buckets.setdefault(tuple(r.get(k,'') for k in keys),[]).append(r)
    for key,rr in buckets.items():
        row=dict(zip(keys,key));row['n_seeds']=len({r['seed'] for r in rr})
        for metric in ['ndcg','recall','auroc','ap','train_seconds']:
            vals=[r[metric] for r in rr if isinstance(r.get(metric),float)]
            row[metric+'_mean']=float(np.mean(vals)) if vals else ''
            row[metric+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else ''
        summary.append(row)
    with (directory/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(f'Saved figures to {target}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    plot(p.parse_args().input)
