"""Wrist-finger correlation of the frozen expert's 20-D actions, from the PCA eigendecomposition.

C = W_robot diag(eigenvalues_robot) W_robot^T is the exact sample covariance (full 20/20 basis).
Usage: python wrist_finger_corr.py s1 <pca_basis.npz> m2 <pca_basis.npz>
Basis files (cluster): ~/data/low-dim-intent/datasets/action-pca-expert-rollout-seed0-job1023537-job1024096/
and action-pca-expert-m2-rollout-seed0-job1033178-job1033179/. Lightweight; run anywhere with numpy.
"""
import sys
import numpy as np
for t, path in zip(sys.argv[1::2], sys.argv[2::2]):
    z=np.load(path); W,l=z['W_robot'],z['eigenvalues_robot']
    C=W@np.diag(l)@W.T; names=list(z['action_names'])
    # consistency: finger block eigenvalues must equal eigenvalues_finger
    ef=np.sort(np.linalg.eigvalsh(C[2:,2:]))[::-1]
    print(f'\n== {t}  names[0:2]={names[:2]}  finger-eig check max rel err={np.max(np.abs(ef-z["eigenvalues_finger"])/z["eigenvalues_finger"].max()):.1e}')
    sd=np.sqrt(np.diag(C)); R=C/np.outer(sd,sd)
    wf=R[:2,2:]; ff=R[2:,2:][np.triu_indices(18,1)]
    print(f'wrist var share of total: {np.diag(C)[:2].sum()/np.trace(C):.3f}  (2/20 = 0.10 if equal)')
    print(f'corr WRJ1-WRJ0: {R[0,1]:+.3f}')
    for i in range(2):
        j=np.argsort(-np.abs(wf[i]))[:3]
        print(f'{names[i]}: mean|r| to fingers {np.abs(wf[i]).mean():.3f}, top: '+', '.join(f'{names[2+k]} {wf[i,k]:+.2f}' for k in j))
    print(f'finger-finger mean|r| {np.abs(ff).mean():.3f}, median {np.median(np.abs(ff)):.3f}')
    # linear R^2 of each wrist action predicted from all 18 fingers
    Cff_inv=np.linalg.inv(C[2:,2:])
    for i in range(2): print(f'R2({names[i]} | 18 fingers) = {C[i,2:]@Cff_inv@C[2:,i]/C[i,i]:.3f}')
    # canonical correlations wrist vs fingers
    A=np.linalg.inv(np.linalg.cholesky(C[:2,:2])); B=np.linalg.inv(np.linalg.cholesky(C[2:,2:]))
    print('canonical corr:', np.round(np.linalg.svd(A@C[:2,2:]@B.T,compute_uv=False),3))
    # fraction of wrist variance captured by hand-k, and loading share
    tot=np.trace(C); fe=z['eigenvalues_finger']; vw=np.diag(C)[:2].sum()
    for k in (2,4,8):
        P=W[:,:k]; cap=np.einsum('ik,k,jk->ij',P,l[:k],P)  # covariance captured by top-k
        wcap=np.diag(cap)[:2].sum()/vw
        print(f'k={k}: hand-k captures {l[:k].sum()/tot:.3f} of total var, {wcap:.3f} of wrist var | '
              f'wrist2-finger-{k-2 if k>2 else 0} (k total nums) captures {(vw+fe[:k-2].sum())/tot:.3f}')
    print('wrist loading share (sum w^2 on wrist) of PC1..8:', np.round((W[:2,:8]**2).sum(0),3))
