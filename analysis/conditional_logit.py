"""Exact patient-conditional logistic likelihood with stable dynamic programming.

This evaluates the same likelihood as statsmodels ConditionalLogit, with no
penalty or likelihood approximation. Patient constants may be centered away.
Numba is an optional local accelerator; no data are exported by this module.
"""
from __future__ import annotations
import os,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

def _enable_numba():
    from numba import njit
    return njit

njit=_enable_numba()

@njit(cache=True)
def conditional_ll_scores(beta,x,y,starts,ends):
    """Return total log likelihood and one score vector per patient."""
    groups=len(starts);p=x.shape[1]
    scores=np.zeros((groups,p));ll=0.0
    for g in range(groups):
        left=starts[g];right=ends[g];n=right-left
        m=int(np.sum(y[left:right]));flip=m>n-m
        if flip:m=n-m
        assert m>0 and m<n
        eta=np.empty(n)
        for i in range(n):
            eta[i]=np.dot(x[left+i],beta)
            if flip:eta[i]=-eta[i]
        eta-=np.max(eta)
        f=np.full((n+1,m+1),-np.inf);b=np.full((n+1,m+1),-np.inf)
        f[0,0]=0.0;b[n,0]=0.0
        for i in range(n):
            f[i+1,0]=0.0
            for k in range(1,min(m,i+1)+1):
                f[i+1,k]=np.logaddexp(f[i,k],f[i,k-1]+eta[i])
        for i in range(n-1,-1,-1):
            b[i,0]=0.0
            for k in range(1,min(m,n-i)+1):
                b[i,k]=np.logaddexp(b[i+1,k],b[i+1,k-1]+eta[i])
        logz=f[n,m];observed=0.0
        for i in range(n):
            yi=y[left+i]
            yf=1.0-yi if flip else yi
            observed+=yf*eta[i]
            acc=-np.inf
            for k in range(max(0,m-1-(n-i-1)),min(m-1,i)+1):
                acc=np.logaddexp(acc,f[i,k]+b[i+1,m-1-k])
            prob=np.exp(eta[i]+acc-logz)
            if flip:prob=1.0-prob
            residual=yi-prob
            for j in range(p):scores[g,j]+=residual*x[left+i,j]
        ll+=observed-logz
    return ll,scores

def information(beta,x,y,starts,ends,step=1e-5):
    """Observed information via central differences of the exact score."""
    p=len(beta);h=np.empty((p,p))
    for j in range(p):
        up=beta.copy();down=beta.copy();up[j]+=step;down[j]-=step
        su=conditional_ll_scores(up,x,y,starts,ends)[1].sum(axis=0)
        sd=conditional_ll_scores(down,x,y,starts,ends)[1].sum(axis=0)
        h[:,j]=-(su-sd)/(2*step)
    return (h+h.T)/2

def fit_exact_conditional(x,y,starts,ends,initial=None,progress=None):
    """Fit a centered/scaled exact CML and calculate patient-score sandwich CIs."""
    x=np.ascontiguousarray(x,dtype=np.float64)
    y=np.ascontiguousarray(y,dtype=np.float64)
    starts=np.asarray(starts,dtype=np.int64);ends=np.asarray(ends,dtype=np.int64)
    # Centering subtracts a constant within patient; conditional likelihood is
    # invariant to this operation and absorbs time-invariant patient terms.
    centered=x.copy()
    for a,b in zip(starts,ends):centered[a:b]-=centered[a:b].mean(axis=0)
    scale=np.sqrt(np.mean(centered**2,axis=0));assert (scale>1e-12).all()
    xs=np.ascontiguousarray(centered/scale)
    ng=len(starts)
    calls=[0]
    def objective(gamma):
        ll,s=conditional_ll_scores(gamma,xs,y,starts,ends)
        calls[0]+=1
        return -ll/ng,-s.sum(axis=0)/ng
    gamma0=np.zeros(x.shape[1]) if initial is None else np.asarray(initial)*scale
    def callback(gamma):
        if progress is not None and calls[0]%20==0:
            f,g=objective(gamma);progress({"function_calls":calls[0],"ll":-f*ng,"max_mean_scaled_score":float(np.max(np.abs(g)))})
    result=minimize(objective,gamma0,method="BFGS",jac=True,callback=callback,options={"gtol":1e-9,"maxiter":500})
    gamma=result.x.copy()
    # Check the actual score and Newton decrement, independent of optimizer's
    # success flag or a threshold on a sample-size-dependent unscaled score.
    for _ in range(3):
        ll,s=conditional_ll_scores(gamma,xs,y,starts,ends);gradient=s.sum(axis=0)
        h=information(gamma,xs,y,starts,ends)
        eigenvalues=np.linalg.eigvalsh(h);assert eigenvalues.min()>1e-8
        step=np.linalg.solve(h,gradient)
        decrement=float(gradient@step)
        if np.max(np.abs(gradient))/ng<1e-10 and decrement<1e-8:break
        trial=gamma+step;trialll=conditional_ll_scores(trial,xs,y,starts,ends)[0]
        assert trialll>=ll-1e-7
        gamma=trial
    ll,s=conditional_ll_scores(gamma,xs,y,starts,ends);gradient=s.sum(axis=0)
    h=information(gamma,xs,y,starts,ends)
    alternate_h=information(gamma,xs,y,starts,ends,step=2e-5)
    assert np.allclose(h,alternate_h,rtol=2e-5,atol=2e-4)
    bread=np.linalg.inv(h)
    cov=bread@(s.T@s)@bread*(ng/(ng-1))
    cov= (cov+cov.T)/2
    beta=gamma/scale
    original_cov=cov/np.outer(scale,scale)
    original_scores=s*scale
    raw_score=original_scores.sum(axis=0)
    decrement=float(gradient@np.linalg.solve(h,gradient))
    assert np.max(np.abs(gradient))/ng<1e-8 and decrement<1e-7
    assert np.linalg.eigvalsh(original_cov).min()>-1e-10
    return {"params":beta,"covariance":original_cov,"ll":ll,"information_scaled":h,"scores":original_scores,
            "nobs":len(y),"n_groups":ng,"n_params":x.shape[1],"centered_scale":scale,
            "optimizer_success":bool(result.success),"optimizer_message":str(result.message),"optimizer_iterations":int(result.nit),"objective_calls":calls[0],
            "score_max_abs":float(np.max(np.abs(raw_score))),"mean_scaled_score_max_abs":float(np.max(np.abs(gradient))/ng),
            "newton_decrement":decrement,"information_min_eigenvalue_scaled":float(np.linalg.eigvalsh(h).min()),"hessian_step_check":True,"converged":True}
