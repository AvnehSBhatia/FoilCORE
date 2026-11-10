import json, sys, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from multiprocessing import Pool
from src.benchmark import load_models
from src.data import AeroDataset, collate_variable_sequences
from src.train_model import prepare_polar_batch
from torch.utils.data import DataLoader, Subset
from tools.paper_numbers import _nf_worker, strided_rows
def main():
    model,stats,cfg,_=load_models('checkpoints/v61/foilcore_best.pt',torch.device('cpu')); model.eval()
    ds=AeroDataset('data/test.csv',stats=stats,training=False)
    rows=strided_rows(len(ds),1024)
    recs=[(ds.df.iloc[i][["coords","alpha","Cl","Cd","Re","mach"]].to_dict(),["xxsmall","xxxlarge"]) for i in rows]
    with Pool(6) as pool: res=pool.starmap(_nf_worker, recs, chunksize=4)
    dl=DataLoader(Subset(ds,rows),batch_size=256,shuffle=False,collate_fn=collate_variable_sequences)
    std=torch.as_tensor(stats['polar_std'][3:5],dtype=torch.float32)
    ours=[]
    with torch.no_grad():
        for b in dl:
            g,p,m=prepare_polar_batch(b,torch.device('cpu')); f,_=model(g,p,m,causal=True)
            e=((f-p[...,3:5]).abs()*std)
            for i in range(g.shape[0]): n=int(m[i].sum()); ours.append(e[i,:n].numpy())
    AL=[];RE=[];MA=[];E_us=[];E_nf={s:[] for s in ("xxsmall","xxxlarge")}
    for (al,cl,cd,o),(rec,_),eu in zip(res,recs,ours):
        n=len(al); AL.append(al); RE.append(np.full(n,rec['Re'])); MA.append(np.full(n,rec['mach'])); E_us.append(eu)
        for s in E_nf: E_nf[s].append(np.stack([np.abs(o[s][0]-cl),np.abs(o[s][1]-cd)],1))
    AL,RE,MA=map(np.concatenate,(AL,RE,MA)); E_us=np.concatenate(E_us); E_nf={s:np.concatenate(v) for s,v in E_nf.items()}
    out={}
    def report(name,mask):
        r={"n":int(mask.sum()),"v61":[float(E_us[mask,0].mean()),float(E_us[mask,1].mean())]}
        for s in E_nf: r[s]=[float(E_nf[s][mask,0].mean()),float(E_nf[s][mask,1].mean())]
        out[name]=r
        print(f"{name:28s} n={r['n']:6d}  v61 Cl {r['v61'][0]:.4f} Cd {r['v61'][1]:.5f} | NF-xxs {r['xxsmall'][0]:.4f} {r['xxsmall'][1]:.5f} | NF-xxxl {r['xxxlarge'][0]:.4f} {r['xxxlarge'][1]:.5f}")
    report("all",np.ones_like(AL,bool))
    for lo,hi in ((-0.01,0.05),(0.05,0.15),(0.15,0.3),(0.3,0.51)): report(f"Mach [{lo:.2f},{hi:.2f})",(MA>=lo)&(MA<hi))
    for lo,hi in ((4e4,1e5),(1e5,3e5),(3e5,1e6),(1e6,3e6),(3e6,2.1e7)): report(f"Re [{lo:.0e},{hi:.0e})",(RE>=lo)&(RE<hi))
    report("Mach<0.05 & Re in [1e5,3e6)",(MA<0.05)&(RE>=1e5)&(RE<3e6))
    report("|alpha|>=12",np.abs(AL)>=12); report("|alpha|<12",np.abs(AL)<12)
    json.dump(out,open('results/paper_final/main/neuralfoil_by_regime.json','w'),indent=1)
if __name__=="__main__": main()
