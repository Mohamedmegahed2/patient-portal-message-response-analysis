from __future__ import annotations
import os
for _key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ.setdefault(_key,"4")
from pathlib import Path
from datetime import datetime,timezone
import gc,hashlib,json,sys,warnings,time
import numpy as np
import pandas as pd
import patsy
import scipy,statsmodels
import statsmodels.formula.api as smf
from scipy.stats import norm
from scipy.special import expit,xlogy,logit
from statsmodels.stats.sandwich_covariance import cov_cluster
from .common import ROOT,OUTPUT_ROOT,WORKING,PRIMARY,MESSAGES,ENCOUNTERS,new_run,finish,require,latest_run

INPUT=PRIMARY
Y="patient_replied_7d"
WINDOWS=[7,14,30,90]
CAPTURE_START=pd.Timestamp("2021-03-01")
FULL90_START=CAPTURE_START+pd.Timedelta(days=90)
CONTROLS=['Age_at_Message','C(Gender)','C(Race, Treatment(reference="White"))','Ethnicity','C(Charlson_group)','C(month_year)']
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def js(path,value):path.write_text(json.dumps(value,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')

def say(message):print(datetime.now().strftime('%H:%M:%S'),message,flush=True)

def signature(d):
    return hashlib.sha256(pd.util.hash_pandas_object(d[['thread_id','DMRN',Y]].astype(str),index=False).to_numpy().tobytes()).hexdigest()

def terms(model,w):
    if model==1:
        a,b=f'log_provider_msgs_past_{w}d',f'log_system_msgs_past_{w}d'
        return [a,b,f'log_patient_initiated_threads_past_{w}d',f'{a}:{b}']
    a,b=f'log_physapp_msgs_past_{w}d',f'log_other_human_msgs_past_{w}d'
    return [a,b,f'log_system_msgs_past_{w}d',f'log_patient_initiated_threads_past_{w}d',f'{a}:{b}']

def formula(model,w,encounter=True):
    ts=terms(model,w)+(['log_encounters_past_7d'] if encounter else [])+CONTROLS
    return f'{Y} ~ '+' + '.join(ts)

def load_data():
    d=pd.read_parquet(INPUT)
    for col in d:
        if isinstance(d[col].dtype,pd.StringDtype):d[col]=d[col].astype(object)
    d['thread_first_time']=pd.to_datetime(d.thread_first_time)
    d['month_year']=d.thread_first_time.dt.to_period('M').astype(str)
    d['Race']=np.select([d.Race.eq('White'),d.Race.eq('Black/African American'),d.Race.eq('Asian')],['White','Black/African American','Asian'],default='Other/Multiple/Unknown')
    d['Race']=pd.Categorical(d.Race,categories=['White','Black/African American','Asian','Other/Multiple/Unknown'])
    d['Ethnicity']=pd.Categorical(d.Ethnicity,categories=['Not Hispanic/Latino','Hispanic/Latino','Unknown/Other'])
    d['Charlson_group']=d.Charlson_group.astype('category')
    assert len(d)==458525 and not d.thread_id.duplicated().any()
    assert d[Y].notna().all() and d[Y].isin([0,1]).all()
    assert d.thread_first_time.between('2021-03-31','2023-11-30 23:59:59.999999999').all()
    for prefix in ['provider_msgs','physapp_msgs','other_human_msgs','system_msgs','patient_initiated_threads']:
        raw=d[[f'{prefix}_past_{w}d' for w in WINDOWS]].to_numpy()
        assert np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
        assert (np.diff(raw,axis=1)>=0).all()
        for w in WINDOWS:assert np.allclose(np.log1p(d[f'{prefix}_past_{w}d']),d[f'log_{prefix}_past_{w}d'],atol=1e-14)
    return d

def audit_counts(d,run):
    say('Independently recounting all four windows from message timestamps and sender metadata')
    m=pd.read_parquet(MESSAGES,columns=['MESSAGE_ID','DMRN','CREATED_TIME','thread_id','Sender_all','is_system_generated'])
    m['CREATED_TIME']=pd.to_datetime(m.CREATED_TIME,errors='coerce')
    m=m.dropna(subset=['DMRN','thread_id','CREATED_TIME']).copy()
    m['DMRN']=m.DMRN.astype(str);m['thread_id']=m.thread_id.astype(str)
    m['sort_id']=pd.to_numeric(m.MESSAGE_ID,errors='coerce')
    is_system=m.Sender_all.eq('System')|m.is_system_generated.fillna(0).astype(int).eq(1)
    m['sender_group']=np.select([m.Sender_all.eq('Patient'),is_system,m.Sender_all.isin(['Physician','APP'])],['Patient','System','PhysAPP'],default='OtherHuman')
    first=m.sort_values(['thread_id','CREATED_TIME','sort_id'],kind='mergesort').groupby('thread_id',as_index=False).first()
    patient_first=first[first.sender_group.eq('Patient')]
    p_times={k:np.sort(g.CREATED_TIME.to_numpy(dtype='datetime64[ns]').view('int64')) for k,g in patient_first.groupby('DMRN',observed=True)}
    message_index=m.groupby('DMRN',observed=True).indices
    counts={(prefix,w):np.zeros(len(d),dtype='int64') for prefix in ['provider_msgs','physapp_msgs','other_human_msgs','system_msgs','patient_initiated_threads'] for w in WINDOWS}
    for pid,indices in d.groupby('DMRN',observed=True).indices.items():
        anchors=d.iloc[indices].thread_first_time.to_numpy(dtype='datetime64[ns]').view('int64')
        g=m.iloc[message_index[str(pid)]];times=g.CREATED_TIME.to_numpy(dtype='datetime64[ns]').view('int64');role=g.sender_group.to_numpy()
        lookup={'physapp_msgs':np.sort(times[role=='PhysAPP']),'other_human_msgs':np.sort(times[role=='OtherHuman']),'system_msgs':np.sort(times[role=='System']),'patient_initiated_threads':p_times.get(str(pid),np.array([],dtype='int64'))}
        lookup['provider_msgs']=np.sort(times[(role=='PhysAPP')|(role=='OtherHuman')])
        for prefix,ts in lookup.items():
            right=np.searchsorted(ts,anchors,side='right')
            for w in WINDOWS:counts[(prefix,w)][indices]=right-np.searchsorted(ts,anchors-w*86400000000000,side='left')
    rows=[]
    for (prefix,w),values in counts.items():
        mismatch=int(np.count_nonzero(values!=d[f'{prefix}_past_{w}d'].to_numpy()))
        rows.append({'exposure':prefix,'window_days':w,'threads':len(d),'mismatched_counts':mismatch,'minimum':int(values.min()),'median':float(np.median(values)),'p90':float(np.quantile(values,.9)),'p99':float(np.quantile(values,.99))})
        assert mismatch==0,(prefix,w,mismatch)
    pd.DataFrame(rows).to_csv(run/'independent_exposure_count_verification.csv',index=False)
    js(run/'exposure_boundaries.json',{'source_messages':len(m),'earliest_recorded_message':str(m.CREATED_TIME.min()),'latest_recorded_message':str(m.CREATED_TIME.max()),'capture_start':str(CAPTURE_START),'message_count_interval':'[anchor-window, anchor]; includes the initiating message and all messages exactly at anchor time','initiating_message_included':True,'thread_gap_days':30,'thread_gap_unchanged_across_models':True,'full_90day_history_from':str(FULL90_START),'early_analytic_threads':int((d.thread_first_time<FULL90_START).sum()),'early_analytic_threads_pct':float(100*(d.thread_first_time<FULL90_START).mean()),'all_20_exposure_count_vectors_reproduced':True,'prior_patient_thread_window':'Matches the indicated message-exposure window, as in original notebook 06; other nonexposure controls unchanged.'})
    say('All 20 raw exposure vectors and log transforms agree with the canonical handoff')

def fit(d,model,w,scope,run):
    key=f'{scope}_M{model}_{w}d'
    f=formula(model,w)
    yframe,xframe=patsy.dmatrices(f,d,return_type='dataframe');assert len(xframe)==len(d)
    n,k=xframe.shape;ng=d.DMRN.nunique();y=yframe[Y].to_numpy()
    say(f'Fitting {key}: {n:,} threads')
    r=smf.logit(f,data=d).fit(disp=False,maxiter=150,tol=1e-10,cov_type='cluster',cov_kwds={'groups':d.DMRN})
    assert r.mle_retvals['converged'] and r.nobs==n
    beta=r.params.loc[xframe.columns].to_numpy()
    cov=r.cov_params().loc[xframe.columns,xframe.columns]
    ll=float(r.llf);converged=bool(r.mle_retvals['converged']);origin={'reused':False}
    X=xframe.to_numpy();pred=expit(X@beta)
    recalculated_ll=float((xlogy(y,pred)+xlogy(1-y,1-pred)).sum())
    assert abs(recalculated_ll-ll)<1e-7
    # Independent patient-cluster sandwich covariance check for every fit.
    bread=np.linalg.inv(X.T@(X*(pred*(1-pred))[:,None]))
    codes,ids=pd.factorize(d.DMRN,sort=False);cluster_scores=np.zeros((len(ids),k))
    np.add.at(cluster_scores,codes,X*(y-pred)[:,None])
    independent=bread@(cluster_scores.T@cluster_scores)@bread*(ng/(ng-1))*((n-1)/(n-k))
    assert np.allclose(independent,cov.to_numpy(),rtol=1e-6,atol=1e-8)
    max_score=float(abs(cluster_scores.sum(axis=0)).max()/n);assert max_score<1e-8
    se=np.sqrt(np.diag(cov));z=norm.ppf(.975)
    c=pd.DataFrame({'fit':key,'scope':scope,'model':model,'window_days':w,'term':xframe.columns,'beta':beta,'cluster_robust_se':se,'OR':np.exp(beta),'CI_lower':np.exp(beta-z*se),'CI_upper':np.exp(beta+z*se),'p_value':2*norm.sf(abs(beta/se)),'threads':n,'patients':ng})
    c.to_csv(run/f'{key}_coefficients.csv',index=False);cov.to_csv(run/f'{key}_patient_clustered_covariance.csv')
    mean=y.mean();llnull=float(xlogy(y,mean).sum()+xlogy(1-y,1-mean).sum())
    fitrow={'fit':key,'scope':scope,'model':model,'window_days':w,'threads':n,'patients':ng,'parameters':k,'events':int(y.sum()),'sample_signature':signature(d),'log_likelihood':ll,'null_log_likelihood':llnull,'AIC':-2*ll+2*k,'BIC':-2*ll+k*np.log(n),'McFadden_R2':1-ll/llnull,'converged':converged,'max_mean_score':max_score,'independent_covariance_max_abs_difference':float(abs(independent-cov.to_numpy()).max()),**origin}
    js(run/f'{key}_diagnostics.json',{'formula':f,**fitrow})
    say('Saved '+key)
    return c,fitrow

def build_tables(coefficients,fits,run):
    def display(model,w,term,scope='full',digits=2):
        r=coefficients[coefficients.model.eq(model)&coefficients.window_days.eq(w)&coefficients.term.eq(term)&coefficients.scope.eq(scope)];assert len(r)==1;r=r.iloc[0]
        return f'{r.OR:.{digits}f} ({r.CI_lower:.{digits}f}–{r.CI_upper:.{digits}f})'
    rows=[]
    for model in [1,2,3]:
        labels=['HCW messages','System messages','Prior patient-initiated threads','HCW × system'] if model==1 else ['Physician/APP messages','Other HCW messages','System messages','Prior patient-initiated threads','Physician/APP × other HCW']
        labels+=['Healthcare encounters (7 days)']
        for j,label in enumerate(labels):
            row={'model':model,'variable':label}
            for w in WINDOWS:
                term=(terms(model,w)+['log_encounters_past_7d'])[j]
                row[f'{w}d_OR_95CI']=display(model,w,term,digits=3 if ':' in term else 2)
            rows.append(row)
    pd.DataFrame(rows).to_csv(run/'Table_S13_window_gradient_estimates.csv',index=False)
    fits.to_csv(run/'Table_S14_window_fit_and_history_coverage.csv',index=False)

    coefficients.loc[coefficients.scope.eq("full") & coefficients.window_days.eq(90)].to_csv(
        run/"Table_S1_90day_coefficients.csv",index=False)


def run_analysis():
    run=new_run("exposure_window_sensitivity",[PRIMARY,MESSAGES])
    d=load_data();audit_counts(d,run)
    estimates=[];rows=[]
    for scope,data in {"full":d,"full90_history":d.loc[d.thread_first_time.ge(FULL90_START)].copy()}.items():
        for model in [1,2,3]:
            sample=data if model==1 else data.loc[data.thread_initiator_group.eq("PhysAPP" if model==2 else "OtherHuman")].copy()
            for w in WINDOWS:
                c,f=fit(sample,model,w,scope,run);estimates.append(c);rows.append(f)
    c=pd.concat(estimates,ignore_index=True);f=pd.DataFrame(rows)
    for (scope,model),g in f.groupby(["scope","model"]):
        assert len(g)==4 and g.sample_signature.nunique()==1 and g.parameters.nunique()==1 and g.threads.nunique()==1
        ref=g.loc[g.window_days.eq(30)].iloc[0]
        f.loc[g.index,"delta_AIC_vs_30d"]=g.AIC-ref.AIC
        f.loc[g.index,"delta_BIC_vs_30d"]=g.BIC-ref.BIC
    c.to_csv(run/"all_model_coefficients.csv",index=False)
    f.to_csv(run/"all_model_fit_statistics.csv",index=False)
    build_tables(c,f,run)
    return finish(run)
