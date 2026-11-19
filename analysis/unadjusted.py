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

from scipy.optimize import root
INPUT=PRIMARY
Y="patient_replied_7d"
EXPECTED={1:(458525,47246,112895),2:(84013,28337,34909),3:(374512,46410,77986)}
PREDICTORS=[('age','Age_at_Message'),('sex','C(Gender, Treatment(reference="Female"))'),('race','C(Race, Treatment(reference="White"))'),('ethnicity','C(Ethnicity, Treatment(reference="Not Hispanic/Latino"))'),('cci','C(Charlson_group, Treatment(reference="0 (None)"))'),('prior_patient','log_patient_initiated_threads_past_30d'),('encounters','log_encounters_past_7d'),('system','log_system_msgs_past_30d')]

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,obj):path.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')

def say(s):print(datetime.now().strftime('%H:%M:%S'),s,flush=True)

def signature(d):return hashlib.sha256(pd.util.hash_pandas_object(d[['thread_id','DMRN',Y]].astype(str),index=False).to_numpy().tobytes()).hexdigest()

def prepare():
    d=pd.read_parquet(INPUT)
    for c in d:
        if isinstance(d[c].dtype,pd.StringDtype):d[c]=d[c].astype(object)
    d['Age_at_Message']=d.Age_at_Message.astype(float)
    d['Gender']=pd.Categorical(d.Gender,categories=['Female','Male'])
    d['Race']=pd.Categorical(np.select([d.Race.eq('White'),d.Race.eq('Black/African American'),d.Race.eq('Asian')],['White','Black/African American','Asian'],default='Other/Multiple/Unknown'),categories=['White','Black/African American','Asian','Other/Multiple/Unknown'])
    d['Ethnicity']=pd.Categorical(d.Ethnicity,categories=['Not Hispanic/Latino','Hispanic/Latino','Unknown/Other'])
    d['Charlson_group']=pd.Categorical(d.Charlson_group,categories=['0 (None)','1-2 (Mild)','3-4 (Moderate)','5+ (Severe)'])
    assert len(d)==458525 and not d.thread_id.duplicated().any()
    assert d[Y].isin([0,1]).all() and d.DMRN.notna().all()
    for prefix in ['provider_msgs','physapp_msgs','other_human_msgs','system_msgs','patient_initiated_threads']:
        assert np.allclose(np.log1p(d[f'{prefix}_past_30d']),d[f'log_{prefix}_past_30d'],atol=1e-14)
    assert np.allclose(np.log1p(d.encounters_past_7d),d.log_encounters_past_7d,atol=1e-14)
    return d

def fit_one(d,model,predictor,rhs,run):
    key=f'M{model}_unadjusted_{predictor}'
    say(f'Fitting {key}: {len(d):,} threads')
    r=smf.logit(f'{Y} ~ {rhs}',data=d).fit(disp=False,maxiter=100,tol=1e-10,cov_type='cluster',cov_kwds={'groups':d.DMRN})
    n,k=r.model.exog.shape;ng=d.DMRN.nunique()
    assert (n,ng,int(r.model.endog.sum()))==EXPECTED[model]
    assert r.mle_retvals['converged'] and r.model.data.row_labels.equals(d.index)
    X=r.model.exog;y=r.model.endog;beta=r.params.to_numpy();p=expit(X@beta)
    assert np.isfinite(beta).all() and np.isfinite(r.cov_params()).all().all()
    # Independent coefficient check: closed-form categorical ORs, score root for continuous.
    if rhs.startswith('C('):
        col={'sex':'Gender','race':'Race','ethnicity':'Ethnicity','cci':'Charlson_group'}[predictor]
        rates=d.groupby(col,observed=True)[Y].mean().reindex(d[col].cat.categories)
        assert rates.between(0,1,inclusive='neither').all()
        expected=np.r_[logit(rates.iloc[0]),logit(rates.iloc[1:].to_numpy())-logit(rates.iloc[0])]
        assert np.allclose(beta,expected,rtol=1e-8,atol=1e-9)
        independent_method='Closed-form log odds from category outcome frequencies'
    else:
        def score(b):return X.T@(y-expit(X@b))/n
        def jac(b):
            pp=expit(X@b);return -(X.T@(X*(pp*(1-pp))[:,None]))/n
        solution=root(score,np.r_[logit(y.mean()),np.zeros(k-1)],jac=jac,tol=1e-10)
        assert solution.success and np.max(np.abs(score(solution.x)))<1e-9
        assert np.allclose(beta,solution.x,rtol=1e-7,atol=1e-8)
        independent_method='Independent root of logistic score, initialized at null'
    # Independently assemble finite-sample-corrected patient-cluster sandwich.
    bread=np.linalg.inv(X.T@(X*(p*(1-p))[:,None]))
    codes,ids=pd.factorize(d.DMRN,sort=False);scores=np.zeros((len(ids),k))
    np.add.at(scores,codes,X*(y-p)[:,None])
    covariance=bread@(scores.T@scores)@bread*(ng/(ng-1))*((n-1)/(n-k))
    observed=r.cov_params().to_numpy()
    assert np.allclose(covariance,observed,rtol=1e-6,atol=1e-8)
    max_score=float(np.abs(scores.sum(axis=0)).max()/n);assert max_score<1e-8
    ll=float((xlogy(y,p)+xlogy(1-y,1-p)).sum());assert abs(ll-r.llf)<1e-7
    se=np.sqrt(np.diag(observed));z=norm.ppf(.975)
    c=pd.DataFrame({'model':model,'predictor':predictor,'term':r.params.index,'beta':beta,'patient_clustered_se':se,'OR':np.exp(beta),'CI_lower':np.exp(beta-z*se),'CI_upper':np.exp(beta+z*se),'p_value':2*norm.sf(abs(beta/se)),'threads':n,'patients':ng})
    c.to_csv(run/f'{key}_coefficients.csv',index=False);r.cov_params().to_csv(run/f'{key}_patient_clustered_covariance.csv')
    dg={'fit':key,'model':model,'predictor':predictor,'formula':r.model.formula,'threads':n,'patients':ng,'events':int(y.sum()),'sample_signature':signature(d),'parameters':k,'converged':True,'log_likelihood':ll,'max_mean_score':max_score,'independent_coefficient_check':independent_method,'independent_coefficient_max_abs_difference':float(abs(beta-(expected if rhs.startswith('C(') else solution.x)).max()),'independent_cluster_covariance_max_abs_difference':float(abs(covariance-observed).max())}
    save(run/f'{key}_diagnostics.json',dg)
    return c,dg

def table_rows():
    rows=[('Age at message (year)','age','Age_at_Message'),('Sex (reference: female)',None,None),('Male','sex','[T.Male]'),('Race (reference: White)',None,None),('Asian','race','[T.Asian]'),('Black or African American','race','[T.Black/African American]'),('Other','race','[T.Other/Multiple/Unknown]'),('Ethnicity (reference: non-Hispanic/Latino)',None,None),('Hispanic/Latino','ethnicity','[T.Hispanic/Latino]'),('Unknown/Other','ethnicity','[T.Unknown/Other]'),('Charlson Comorbidity Index (reference: 0)',None,None),('1–2 (mild)','cci','[T.1-2 (Mild)]'),('3–4 (moderate)','cci','[T.3-4 (Moderate)]'),('≥5 (severe)','cci','[T.5+ (Severe)]'),('Patient-initiated messaging (log, past 30 days)','prior_patient','log_patient_initiated_threads_past_30d'),('Recent healthcare encounters (log, past 7 days)','encounters','log_encounters_past_7d'),('Message exposures (log, past 30 days)',None,None),('System','system','log_system_msgs_past_30d'),('HCW','hcw','log_provider_msgs_past_30d'),('Physician/APP','physapp','log_physapp_msgs_past_30d'),('Other HCW','other_hcw','log_other_human_msgs_past_30d')]
    return rows

def build_table(c,run):
    rows=[];checks=[]
    for label,predictor,term in table_rows():
        row={'Variable':label}
        for model in [1,2,3]:
            column=f'Model {model} unadjusted OR (95% CI)'
            if predictor is None:row[column]='';continue
            x=c[c.model.eq(model)&c.predictor.eq(predictor)&c.term.str.endswith(term)]
            if not len(x):
                assert (model==1 and predictor in ['physapp','other_hcw']) or (model in [2,3] and predictor=='hcw')
                row[column]='—';continue
            assert len(x)==1;r=x.iloc[0]
            row[column]=f'{r.OR:.2f} ({r.CI_lower:.2f}–{r.CI_upper:.2f})'
            checks.append({'label':label,'model':model,'predictor':predictor,'term':r.term,'display':row[column]})
        rows.append(row)
    t=pd.DataFrame(rows);assert len(checks)==44
    t.to_csv(run/'Table_S15_unadjusted_ORs.tsv',sep='\t',index=False)
    t.to_csv(run/'Table_S15_unadjusted_ORs.csv',index=False)
    pd.DataFrame(checks).to_csv(run/'Table_S15_display_verification.csv',index=False)
    return t

def run_analysis():
    run=new_run("unadjusted_associations",[PRIMARY])
    d=prepare();cs=[];ds=[];samples=[]
    for model in [1,2,3]:
        sample=d if model==1 else d.loc[d.thread_initiator_group.eq("PhysAPP" if model==2 else "OtherHuman")].copy()
        assert (len(sample),sample.DMRN.nunique(),int(sample[Y].sum()))==EXPECTED[model]
        specs=PREDICTORS+([("hcw","log_provider_msgs_past_30d")] if model==1 else [("physapp","log_physapp_msgs_past_30d"),("other_hcw","log_other_human_msgs_past_30d")])
        for predictor,rhs in specs:
            c,z=fit_one(sample,model,predictor,rhs,run);cs.append(c);ds.append(z)
        samples.append({"model":model,"threads":len(sample),"patients":sample.DMRN.nunique(),"reply_events":int(sample[Y].sum()),"sample_signature":signature(sample)})
    c=pd.concat(cs,ignore_index=True);dg=pd.DataFrame(ds)
    assert len(dg)==29 and len(c.loc[c.term.ne("Intercept")])==44 and dg.converged.all()
    c.to_csv(run/"all_unadjusted_coefficients.csv",index=False)
    dg.to_csv(run/"all_model_diagnostics.csv",index=False)
    pd.DataFrame(samples).to_csv(run/"primary_samples.csv",index=False)
    build_table(c,run)
    return finish(run)
