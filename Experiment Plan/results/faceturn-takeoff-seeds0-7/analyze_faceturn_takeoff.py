"""Face-turn E1 take-off counts at iteration 5000, seeds 0-7 (1153680/1153770/1157733/1157824).
Take-off = turns per episode > TH at 5000. Fisher exact test per rate and pooled."""
import csv,glob,os,re,sys
from scipy.stats import fisher_exact
PE=31.77;TH=10
R={}
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),"curves","*.csv"))):
    m=re.search(r"hidden-(goal-yaw|hand-8)(?:-hold(\d+))?-seed(\d)",f)
    it,rate,s=m.group(1),"10Hz" if m.group(2) else "60Hz",int(m.group(3))
    rows=list(csv.DictReader(open(f)))
    y=[float(r["mean_successes_per_episode"]) for r in rows]; fl=[float(r["fall_fraction"]) for r in rows]
    its=[int(r["iteration"]) for r in rows]
    t=next((its[k] for k in range(len(y)) if y[k]>TH),None)
    R[(it,rate,s)]=(y[-1],max(y),t,fl[-1],sum(y[-5:])/5, y[-5])
for rate in ["60Hz","10Hz"]:
  for it in ["goal-yaw","hand-8"]:
    print(f"\n{it} {rate}: seed final(/PE) max first>{TH} fall@5000 last5mean v4200")
    for s in range(8):
        f,mx,t,fl,l5,v42=R[(it,rate,s)]
        print(f"  s{s} {f:6.2f} ({f/PE:.2f}) max {mx:6.2f} first {t} fall {fl:.2f} last5 {l5:5.2f} @4200 {v42:5.2f}")
    print("  mean final/PE %.3f"%(sum(R[(it,rate,s)][0] for s in range(8))/8/PE))
k=lambda it,rate:sum(R[(it,rate,s)][0]>TH for s in range(8))
for rate in ["60Hz","10Hz"]:
    a,b=k("goal-yaw",rate),k("hand-8",rate)
    print(rate,"take-off goal-yaw",a,"/8 hand-8",b,"/8 Fisher two-sided p=%.4f one-sided p=%.4f"%(fisher_exact([[a,8-a],[b,8-b]])[1],fisher_exact([[a,8-a],[b,8-b]],alternative="greater")[1]))
a=k("goal-yaw","60Hz")+k("goal-yaw","10Hz");b=k("hand-8","60Hz")+k("hand-8","10Hz")
print("pooled",a,"/16",b,"/16 two-sided p=%.4f one-sided %.4f"%(fisher_exact([[a,16-a],[b,16-b]])[1],fisher_exact([[a,16-a],[b,16-b]],alternative="greater")[1]))
for th in [5,15,20]:
    c={(i,r):sum(R[(i,r,s)][0]>th for s in range(8)) for i in ["goal-yaw","hand-8"] for r in ["60Hz","10Hz"]}
    print("threshold",th,c)
