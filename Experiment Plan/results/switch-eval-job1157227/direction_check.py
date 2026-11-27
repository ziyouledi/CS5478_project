import sys, numpy as np
TURN = 2  # placeholder, overwritten below
def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi
for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True); tr = d["trace"]; c = {n: i for i, n in enumerate(d["columns"])}
    T, N, _ = tr.shape
    kinds = np.unique(tr[:, :, c["kind"]]); 
    out = {}
    for e in range(N):
        x = tr[:, e]; ps = x[:, c["phase_steps"]]; st = x[:, c["phase_start_hinge_rad"]]; k = x[:, c["kind"]]
        brk = np.r_[0, np.where((ps[1:] < ps[:-1]) | (st[1:] != st[:-1]) | (k[1:] != k[:-1]))[0] + 1, T]
        for a, b in zip(brk[:-1], brk[1:]):
            seg = x[a:b]; kk = int(seg[0, c["kind"]])
            if kk < 0: continue
            disp = np.degrees(wrap(seg[:, c["hinge_pos_rad"]] - seg[0, c["phase_start_hinge_rad"]]))
            err0 = seg[0, c["hinge_error_rad"]] - np.radians(disp[0])  # error at phase start
            err0 = seg[0, c["phase_start_hinge_rad"]] - (seg[0, c["hinge_pos_rad"]] - seg[0, c["hinge_error_rad"]])
            err0 = np.degrees(wrap(err0))
            sgn = -np.sign(err0) if abs(err0) > 45 else 0
            at_goal_end = abs(np.degrees(seg[-1, c["hinge_error_rad"]])) < 5.7
            out.setdefault((kk, sgn), []).append((np.max(disp*sgn) if sgn else 0, np.max(-disp*sgn) if sgn else 0, np.max(np.abs(disp)), at_goal_end, b - a, b == T))
    print("==", path)
    for (kk, sgn), rows in sorted(out.items()):
        r = np.array(rows, dtype=float)
        tow, awy, absd, atg, ln, openend = r.T
        s = f"kind={kk} dir={'goal+90' if sgn>0 else 'goal-90' if sgn<0 else 'none'} phases={len(r)}"
        # note: sgn=-sign(err0); err0=-90 (goal +90) -> sgn=+1
        if sgn:
            s += f" toward med={np.median(tow):.1f} p90={np.percentile(tow,90):.1f} >10:{(tow>10).mean():.3f} >45:{(tow>45).mean():.3f} away med={np.median(awy):.1f} >45:{(awy>45).mean():.3f} at_goal_end={atg.mean():.3f} open_at_T={openend.mean():.3f}"
        else:
            s += f" |hinge move| med={np.median(absd):.1f} p90={np.percentile(absd,90):.1f} >45:{(absd>45).mean():.3f}"
        print(s)
