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

from statsmodels.discrete.conditional_models import ConditionalLogit
INPUT=PRIMARY
Y="patient_replied_7d"
ADJUST=['Age_at_Message','C(Gender)','C(Race, Treatment(reference="White"))','Ethnicity','C(Charlson_group)','C(month_year)']
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def savejson(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def announce(message):print(datetime.now().strftime('%H:%M:%S'),message,flush=True)

def terms(model,period):
    suf=f'past_{period}d'
    if model==1:
        a,b=f'log_provider_msgs_{suf}',f'log_system_msgs_{suf}'
        return [a,b,f'log_patient_initiated_threads_{suf}',f'{a}:{b}']
    a,b=f'log_physapp_msgs_{suf}',f'log_other_human_msgs_{suf}'
    return [a,b,f'log_system_msgs_{suf}',f'log_patient_initiated_threads_{suf}',f'{a}:{b}']

def formula(model,period,encounter=7,initiator=False):
    ts=terms(model,period)+[f'log_encounters_past_{encounter}d']+ADJUST
    if initiator:ts+=['C(thread_initiator_group, Treatment(reference="OtherHuman"))']
    return f'{Y} ~ '+' + '.join(ts)

def prepare():
    d=pd.read_parquet(INPUT)
    # Patsy 1.0.1 requires Python object or category instead of pandas StringDtype.
    for column in d:
        if isinstance(d[column].dtype,pd.StringDtype):d[column]=d[column].astype(object)
    assert not d.thread_id.duplicated().any() and len(d)==458525
    d['thread_first_time']=pd.to_datetime(d.thread_first_time)
    d['month_year']=d.thread_first_time.dt.to_period('M').astype(str)
    d['Race']=np.select([d.Race.eq('White'),d.Race.eq('Black/African American'),d.Race.eq('Asian')],['White','Black/African American','Asian'],default='Other/Multiple/Unknown')
    d['Race']=pd.Categorical(d.Race,categories=['White','Black/African American','Asian','Other/Multiple/Unknown'])
    d['Ethnicity']=pd.Categorical(d.Ethnicity,categories=['Not Hispanic/Latino','Hispanic/Latino','Unknown/Other'])
    d['Charlson_group']=d.Charlson_group.astype('category')
    assert d[Y].notna().all() and d[Y].isin([0,1]).all()
    assert d.thread_first_time.between('2021-03-31','2023-11-30 23:59:59.999999999').all()
    return d

def audit_encounters(d,run):
    e=pd.read_parquet(ENCOUNTERS,columns=['DMRN','ENCOUNTER_START','SOURCE','TYPE_GROUP','visit_type_grouped'])
    e['ENCOUNTER_START']=pd.to_datetime(e.ENCOUNTER_START)
    assert e[['DMRN','ENCOUNTER_START']].notna().all().all()
    timeline={str(k):np.sort(v.ENCOUNTER_START.to_numpy(dtype='datetime64[ns]').astype('int64')) for k,v in e.groupby('DMRN',observed=True)}
    counts={w:np.zeros(len(d),dtype='int64') for w in [7,30]}
    for key,indices in d.groupby('DMRN',observed=True).indices.items():
        ts=d.iloc[indices].thread_first_time.to_numpy(dtype='datetime64[ns]').astype('int64')
        events=timeline.get(str(key),np.array([],dtype='int64'))
        right=np.searchsorted(events,ts,side='left')
        for w in counts:counts[w][indices]=right-np.searchsorted(events,ts-w*86400000000000,side='left')
    summary=[]
    for w,c in counts.items():
        assert np.array_equal(c,d[f'encounters_past_{w}d'])
        assert np.allclose(np.log1p(c),d[f'log_encounters_past_{w}d'],atol=1e-14)
        summary.append({'encounter_window_days':w,'threads':len(d),'threads_with_encounter':int((c>0).sum()),'zero':int((c==0).sum()),'one':int((c==1).sum()),'two':int((c==2).sum()),'three':int((c==3).sum()),'four_or_more':int((c>=4).sum()),'total_counted_instances':int(c.sum()),'median':float(np.median(c)),'p75':float(np.quantile(c,.75)),'maximum':int(c.max()),'count_mismatches':0,'log_count_mismatches':0})
    pd.DataFrame(summary).to_csv(run/'encounter_count_verification.csv',index=False)
    e.groupby(['SOURCE','TYPE_GROUP','visit_type_grouped'],observed=True).size().rename('records').reset_index().to_csv(run/'encounter_source_summary.csv',index=False)
    ids=set(d.DMRN.astype(str));eid=set(e.DMRN.astype(str))
    a={'source_records':len(e),'source_patients':e.DMRN.nunique(),'source_start':str(e.ENCOUNTER_START.min()),'source_end':str(e.ENCOUNTER_START.max()),'analytic_patients_with_any_encounter_record_in_source_period':len(ids&eid),'analytic_patients_without_encounter_record_in_source_period':len(ids-eid),'source_sha256':sha(ENCOUNTERS),'linkage':'Exact shared deidentified DMRN; no fuzzy matching','window':'[thread initiation minus window, thread initiation); anchor-time encounter excluded','clinic_date_resolution':'Clinic CONTACT_DATE floored to day; represented at midnight. ED and hospital admission timestamps retained.','zero_convention':'No recorded event in window gives zero; log1p(0)=0.','deduplication':'Original notebook 03 rules retained: clinic patient/date/type/provider/department; hospital patient/admission/discharge/type; ED patient/admission, with hospital-ED fallback deduplicated against ED admission minute.'}
    savejson(run/'encounter_provenance.json',a)
    announce('Encounter counts independently matched exactly at both 7 and 30 days')

def coef_table(params,cov,key,estimator,model,period,encounter,n,ng):
    beta=np.asarray(params);se=np.sqrt(np.diag(cov));z=norm.ppf(.975)
    return pd.DataFrame({'fit':key,'estimator':estimator,'model':model,'exposure_days':period,'encounter_days':encounter,'term':params.index,'beta':beta,'cluster_robust_se':se,'OR':np.exp(beta),'CI_lower':np.exp(beta-z*se),'CI_upper':np.exp(beta+z*se),'p_value':2*norm.sf(abs(beta/se)),'threads':n,'patients':ng})

def pooled(d,model,period,encounter,key,run,initiator=False):
    target=run/f'{key}_coefficients.csv'
    if target.exists():
        announce('Reusing completed '+key);return pd.read_csv(target)
    f=formula(model,period,encounter,initiator)
    announce(f'Fitting {key}: {len(d):,} threads')
    r=smf.logit(f,data=d).fit(disp=False,cov_type='cluster',cov_kwds={'groups':d.DMRN},maxiter=150,tol=1e-10)
    assert r.mle_retvals['converged'] and r.nobs==len(d)
    assert np.isfinite(r.params).all() and np.isfinite(r.cov_params()).all().all()
    t=coef_table(r.params,r.cov_params(),key,'Pooled logistic',model,period,encounter,len(d),d.DMRN.nunique())
    t.to_csv(target,index=False);r.cov_params().to_csv(run/f'{key}_patient_clustered_covariance.csv')
    savejson(run/f'{key}_diagnostics.json',{'formula':f,'threads':len(d),'patients':d.DMRN.nunique(),'converged':bool(r.mle_retvals['converged']),'max_mean_score':float(abs(r.model.score(r.params)).max()/len(d)),'ll':r.llf,'fit_info':r.mle_retvals})
    announce('Saved '+key)
    return t

def conditional_sample(d,model,period):
    ts=terms(model,period)
    required=[Y,'DMRN','month_year','Age_at_Message','Gender','Race','Ethnicity','Charlson_group','log_encounters_past_7d']+[t for t in ts if ':' not in t]
    complete=d.dropna(subset=required).copy()
    assert len(complete)==len(d)
    g=complete.groupby('DMRN',observed=True)[Y].agg(['size','sum','nunique'])
    retained=g.index[g['nunique'].eq(2)]
    c=complete.loc[complete.DMRN.isin(retained)].sort_values(['DMRN','thread_first_time','thread_id'],kind='mergesort').copy()
    counts=c.groupby('DMRN',sort=False,observed=True).size().to_numpy()
    ends=np.cumsum(counts);starts=ends-counts
    s={'model':model,'exposure_days':period,'full_threads':len(complete),'full_patients':len(g),'retained_threads':len(c),'retained_patients':len(counts),'retained_threads_pct':100*len(c)/len(complete),'patients_all_no_reply':int(g['sum'].eq(0).sum()),'patients_all_reply':int(g['sum'].eq(g['size']).sum()),'excluded_patients':int(g['nunique'].eq(1).sum()),'excluded_threads':len(complete)-len(c)}
    return c,starts,ends,s

def conditional_design(c,model,period):
    principal=terms(model,period)+['log_encounters_past_7d']
    x=patsy.dmatrix(' + '.join(principal+['C(Charlson_group)','C(month_year)']),data=c,return_type='dataframe').drop(columns='Intercept')
    ranges=(x-x.groupby(c.DMRN,observed=True).transform('mean')).abs().max()
    dropped=ranges.index[ranges<=1e-12].tolist()
    x=x.drop(columns=dropped)
    assert all(t in x for t in principal)
    for v in ['Age_at_Message','Gender','Race','Ethnicity','Charlson_group']:
        assert c.groupby('DMRN',observed=True)[v].nunique().max()==1
    return x,dropped,principal

def verify_real_kernel(x,y,starts,ends,beta):
    from .conditional_logit import conditional_ll_scores
    # Match statsmodels directly on heterogeneous real groups, including the
    # largest patient group, high event-count groups, and complement cases.
    sizes=ends-starts
    selected=np.unique(np.r_[np.arange(min(20,len(starts))),np.argmax(sizes),np.argsort(sizes)[-5:]])
    xx=[];yy=[];gg=[];ss=[];ee=[];pos=0
    for g in selected:
        a,b=starts[g],ends[g];ss.append(pos);xx.append(x[a:b]);yy.append(y[a:b]);gg.extend([g]*(b-a));pos+=b-a;ee.append(pos)
    xx=np.ascontiguousarray(np.vstack(xx));yy=np.concatenate(yy);gg=np.array(gg)
    for trial in [np.zeros(x.shape[1]),beta]:
        l,s=conditional_ll_scores(trial,xx,yy,np.array(ss),np.array(ee))
        sm=ConditionalLogit(yy,xx,groups=gg)
        assert np.allclose(l,sm.loglike(trial),rtol=1e-10,atol=1e-8)
        assert np.allclose(s.sum(0),sm.score(trial),rtol=1e-7,atol=2e-7)
    return {'real_patient_groups_checked':len(selected),'largest_group_included':int(sizes.max()),'statsmodels_likelihood_and_score_agree_at_zero_and_fitted_parameters':True}

def conditional(c,starts,ends,model,period,run):
    key=f'M{model}_{period}d_conditional'
    target=run/f'{key}_coefficients.csv'
    if target.exists():announce('Reusing completed '+key);return pd.read_csv(target)
    from .conditional_logit import fit_exact_conditional
    x,dropped,principal=conditional_design(c,model,period)
    y=c[Y].to_numpy(dtype=float);xx=np.ascontiguousarray(x.to_numpy(dtype=float))
    announce(f'Fitting exact {key}: {len(c):,} threads / {len(starts):,} patients / {x.shape[1]} terms')
    old=pd.read_csv(run/f'M{model}_{period}d_matched_pooled_coefficients.csv').set_index('term')
    initial=old.loc[x.columns,'beta'].to_numpy()
    before=time.monotonic()
    r=fit_exact_conditional(xx,y,starts,ends,initial=initial,progress=lambda a:announce(f'{key} {a}'))
    verification=verify_real_kernel(xx,y,starts,ends,r['params'])
    t=coef_table(pd.Series(r['params'],index=x.columns),r['covariance'],key,'Conditional patient fixed effects',model,period,7,len(c),len(starts))
    t.to_csv(target,index=False)
    pd.DataFrame(r['covariance'],index=x.columns,columns=x.columns).to_csv(run/f'{key}_patient_clustered_covariance.csv')
    diagnostic={k:v for k,v in r.items() if k not in ['params','covariance','information_scaled','scores','centered_scale']}
    diagnostic.update({'elapsed_seconds':time.monotonic()-before,'design_columns':list(x.columns),'dropped_no_within_patient_variation':dropped,'absorbed_patient_constants':['Age_at_Message','Gender','Race','Ethnicity','Charlson_group'],'principal_terms':principal,'estimator':'Exact conditional logistic likelihood; stable log-space dynamic programming, no penalty or approximation','covariance':'Patient-score sandwich, observed information, G/(G-1) finite-group correction; same convention as notebook 08','validation':verification})
    savejson(run/f'{key}_diagnostics.json',diagnostic)
    announce('Saved '+key)
    return t

def create_tables(run, fixed_effects=False):
    coefficients=pd.concat([pd.read_csv(p) for p in sorted(run.glob('*_coefficients.csv')) if p.name!='all_model_coefficients.csv'],ignore_index=True)
    coefficients.to_csv(run/'all_model_coefficients.csv',index=False)
    def val(key,term,digits=2):
        q=coefficients.loc[coefficients.fit.eq(key)&coefficients.term.eq(term)]
        assert len(q)==1,(key,term)
        r=q.iloc[0];return f'{r.OR:.{digits}f} ({r.CI_lower:.{digits}f}–{r.CI_upper:.{digits}f})'
    labels={1:['HCW messages','System messages','Prior patient-initiated threads','HCW × system','Healthcare encounters (7 days)'],2:['Physician/APP messages','Other HCW messages','System messages','Prior patient-initiated threads','Physician/APP × other HCW','Healthcare encounters (7 days)']}
    if fixed_effects:
        rows=[]
        for model in [1,2]:
            for period in [30,90]:
                ts=terms(model,period)+['log_encounters_past_7d']
                for term,label in zip(ts,labels[model]):
                    digits=3 if ':' in term else 2
                    rows.append({'model':model,'exposure_days':period,'variable':label,'pooled_full_OR_95CI':val(f'M{model}_{period}d_primary',term,digits),'pooled_retained_sample_OR_95CI':val(f'M{model}_{period}d_matched_pooled',term,digits),'conditional_patient_FE_OR_95CI':val(f'M{model}_{period}d_conditional',term,digits)})
        pd.DataFrame(rows).to_csv(run/'Table_S7_revised.csv',index=False)

    else:
        rows=[]
        for model in [1,2,3]:
            ts=terms(model,30)+['log_encounters_past_7d']
            ls=labels[1] if model==1 else labels[2]
            for term,label in zip(ts,ls):
                rows.append({'panel':'A. Thirty-day encounter adjustment','model':model,'variable':label.replace('(7 days)','(7/30 days)'),'primary_7day_encounters_OR_95CI':val(f'M{model}_30d_primary',term),'sensitivity_OR_95CI':val(f'M{model}_30d_encounter30',term.replace('log_encounters_past_7d','log_encounters_past_30d'))})
        for term,label in zip(terms(1,30)+['log_encounters_past_7d'],labels[1]):
            rows.append({'panel':'B. Model 1 adjusted for thread initiator type','model':1,'variable':label,'primary_7day_encounters_OR_95CI':val('M1_30d_primary',term),'sensitivity_OR_95CI':val('M1_30d_initiator',term)})
        indicator='C(thread_initiator_group, Treatment(reference="OtherHuman"))[T.PhysAPP]'
        rows.append({'panel':'B. Model 1 adjusted for thread initiator type','model':1,'variable':'Physician/APP initiator (vs other HCW)','primary_7day_encounters_OR_95CI':'Not included','sensitivity_OR_95CI':val('M1_30d_initiator',indicator)})
        pd.DataFrame(rows).to_csv(run/'Table_S12_encounter_and_initiator_sensitivity.csv',index=False)

def run_fixed_effects():
    run=new_run("patient_fixed_effects",[PRIMARY])
    d=prepare()
    samples={1:d,2:d.loc[d.thread_initiator_group.eq("PhysAPP")].copy()}
    flow=[]
    for model in [2,1]:
        for period in [30,90]:
            pooled(samples[model],model,period,7,f"M{model}_{period}d_primary",run)
            c,starts,ends,summary=conditional_sample(samples[model],model,period)
            flow.append(summary)
            pooled(c,model,period,7,f"M{model}_{period}d_matched_pooled",run)
            conditional(c,starts,ends,model,period,run)
    pd.DataFrame(flow).to_csv(run/"conditional_sample_flow.csv",index=False)
    create_tables(run,fixed_effects=True)
    for model in [1,2]:
        for period in [30,90]:
            assert json.loads((run/f"M{model}_{period}d_conditional_diagnostics.json").read_text())["converged"]
    return finish(run)

def run_encounters():
    run=new_run("encounter_sensitivity",[PRIMARY,ENCOUNTERS])
    d=prepare();audit_encounters(d,run)
    for model in [1,2,3]:
        sample=d if model==1 else d.loc[d.thread_initiator_group.eq("PhysAPP" if model==2 else "OtherHuman")].copy()
        pooled(sample,model,30,7,f"M{model}_30d_primary",run)
        pooled(sample,model,30,30,f"M{model}_30d_encounter30",run)
    pooled(d,1,30,7,"M1_30d_initiator",run,initiator=True)
    create_tables(run)
    return finish(run)
