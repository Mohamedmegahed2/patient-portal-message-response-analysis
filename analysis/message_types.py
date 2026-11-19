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

RARE_MINIMUM=100
RARE_LABEL="Other recorded message types (pooled)"
TYPE_REFERENCE="User Message"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def save(p,value):
    Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")

def load_analysis():
    a=pd.read_parquet(PRIMARY)
    a["Age65"]=pd.Categorical(a.Age65,categories=["<65","≥65"])
    a["Gender"]=a.Gender.astype("category")
    a["Ethnicity"]=pd.Categorical(a.Ethnicity,categories=["Not Hispanic/Latino","Hispanic/Latino","Unknown/Other"])
    a["Charlson_group"]=a.Charlson_group.astype("category")
    a["Race"]=pd.Categorical(np.select([a.Race.eq("White"),a.Race.eq("Black/African American"),a.Race.eq("Asian")],["White","Black/African American","Asian"],default="Other/Multiple/Unknown"),categories=["White","Black/African American","Asian","Other/Multiple/Unknown"])
    a["month_year"]=a.thread_first_time.dt.to_period("M").astype(str)
    # Original notebook 06 date restriction; canonical handoff already ends Nov 30.
    a=a.loc[a.Age65.notna()&a.thread_first_time.between(pd.Timestamp("2021-03-31"),pd.Timestamp("2023-12-01"))].copy()
    assert len(a)==458525 and a.DMRN.nunique()==47246
    m=pd.read_parquet(MESSAGES,columns=["MESSAGE_ID","MESG_TYPE","SENDER_CATEGORY","Sender_all"])
    assert m.MESSAGE_ID.is_unique and len(m)==3237639
    corpus=m.groupby("MESG_TYPE",observed=True,dropna=False).size().rename("messages").reset_index()
    a=a.merge(m,left_on="thread_first_message_id",right_on="MESSAGE_ID",how="left",validate="one_to_one",indicator=True)
    assert a._merge.eq("both").all()
    a["initiating_message_type"]=a.MESG_TYPE.astype("string").str.strip()
    assert a.initiating_message_type.notna().all() and a.initiating_message_type.ne("").all()
    a["initiator_structured_role"]=a.SENDER_CATEGORY.astype("string").str.strip().replace("",pd.NA).fillna("Unknown")
    assert a.Sender_all.isin(["Physician","APP"]).eq(a.thread_initiator_group.eq("PhysAPP")).all()
    a["descriptive_sender_category"]=np.where(a.thread_initiator_group.eq("PhysAPP"),"Physician/APP",a.initiator_structured_role)
    keep=["thread_id","DMRN","thread_first_time","thread_initiator_group","patient_replied_7d","Age_at_Message","Age65","Gender","Race","Ethnicity","Charlson_group","month_year","initiating_message_type","initiator_structured_role","descriptive_sender_category",
          "log_physapp_msgs_past_30d","log_other_human_msgs_past_30d","log_system_msgs_past_30d","log_patient_initiated_threads_past_30d","log_encounters_past_7d"]
    return a[keep].copy(),corpus

def descriptions(a,corpus,run):
    corpus.to_csv(run/"full_corpus_recorded_message_types.csv",index=False)
    grouped=a.groupby(["thread_initiator_group","initiating_message_type"],observed=True).agg(threads=("thread_id","size"),replies_7d=("patient_replied_7d","sum")).reset_index()
    totals=a.groupby("thread_initiator_group",observed=True).size()
    grouped["percent_within_sender_category"]=100*grouped.threads/grouped.thread_initiator_group.map(totals)
    grouped["reply_percent"]=100*grouped.replies_7d/grouped.threads
    grouped["no_reply_threads"]=grouped.threads-grouped.replies_7d
    grouped["zero_reply_type"]=grouped.replies_7d.eq(0)
    grouped["pooled_for_adjustment"]=grouped.threads.lt(RARE_MINIMUM)
    grouped.to_csv(run/"initiating_type_by_sender_category.csv",index=False)
    granular=a.groupby(["descriptive_sender_category","initiating_message_type"],observed=True).agg(threads=("thread_id","size"),replies_7d=("patient_replied_7d","sum")).reset_index()
    granular["reply_percent"]=100*granular.replies_7d/granular.threads
    granular.to_csv(run/"initiating_type_by_detailed_sender_role.csv",index=False)
    roles=a.loc[a.thread_initiator_group.eq("OtherHuman")].groupby("initiator_structured_role",observed=True).agg(threads=("thread_id","size"),replies=("patient_replied_7d","sum")).reset_index().rename(columns={"initiator_structured_role":"role"})
    roles["reply_percent"]=100*roles.replies/roles.threads
    roles.to_csv(run/"table_s9_role_totals_reproduced.csv",index=False)
    user=a.loc[a.initiating_message_type.eq(TYPE_REFERENCE)].groupby("descriptive_sender_category",observed=True).agg(threads=("thread_id","size"),replies_7d=("patient_replied_7d","sum")).reset_index()
    user["reply_percent"]=100*user.replies_7d/user.threads
    user.to_csv(run/"within_user_message_role_descriptives.csv",index=False)
    records=[]
    for group in ["PhysAPP","OtherHuman"]:
        d=a.loc[a.thread_initiator_group.eq(group)]
        counts=d.initiating_message_type.value_counts()
        raw=grouped.loc[grouped.thread_initiator_group.eq(group)]
        records.append({"sender_category":group,"n_threads":len(d),"n_recorded_types":len(counts),"user_message_threads":int(counts[TYPE_REFERENCE]),"user_message_percent":100*counts[TYPE_REFERENCE]/len(d),"raw_zero_reply_types":raw.loc[raw.zero_reply_type,"initiating_message_type"].tolist(),"rare_type_labels":counts.loc[counts.lt(RARE_MINIMUM)].index.tolist(),"rare_type_threads":int(counts.loc[counts.lt(RARE_MINIMUM)].sum())})
    save(run/"metadata_and_estimability_audit.json",{"all_initiating_messages_linked":True,"missing_initiating_type":0,"field":"MESG_TYPE","field_provenance":"Source EHR type labels retained by notebook 01 read_message_file; whitespace stripped, not inferred from body text or filenames.","full_corpus_types":len(corpus),"analytic_types":a.initiating_message_type.nunique(),"by_model_sample":records,"raw_full_type_logit_note":"Some observed raw types have zero replies, so a finite unpenalized MLE for every separate type indicator is unavailable. No unpooled separated fit is reported.","frequency_pooling_rule":f"Within each model, retain types with at least {RARE_MINIMUM} initiating threads and pool smaller types; rule uses frequency, not estimated associations.","interpretation":"User Message is a broad recorded category; this metadata does not identify clinical purpose or whether a reply was expected."})
    print(grouped.to_string(index=False),flush=True)
    return grouped

def grouped_type(d,number,run):
    counts=d.initiating_message_type.value_counts()
    mapping={str(k):(str(k) if n>=RARE_MINIMUM else RARE_LABEL) for k,n in counts.items()}
    d=d.copy()
    d["model_message_type"]=pd.Categorical(d.initiating_message_type.map(mapping),categories=[TYPE_REFERENCE]+sorted(set(mapping.values())-{TYPE_REFERENCE}))
    cells=d.groupby("model_message_type",observed=True).patient_replied_7d.agg(["size","sum"])
    assert cells["sum"].gt(0).all() and cells["sum"].lt(cells["size"]).all()
    pd.DataFrame([{"model":number,"recorded_type":k,"model_type":v,"threads":int(counts[k]),"pooled":k!=v} for k,v in mapping.items()]).to_csv(run/f"model_{number}_type_pooling_map.csv",index=False)
    cells.reset_index().rename(columns={"size":"threads","sum":"replies_7d"}).to_csv(run/f"model_{number}_pooled_type_cells.csv",index=False)
    return d

def fit(d,number,specification,formula,run):
    need=["DMRN","patient_replied_7d","Age_at_Message","Gender","Race","Ethnicity","Charlson_group","month_year","log_physapp_msgs_past_30d","log_other_human_msgs_past_30d","log_system_msgs_past_30d","log_patient_initiated_threads_past_30d","log_encounters_past_7d"]
    if specification=="type_adjusted":need.append("model_message_type")
    source_n=len(d);d=d.dropna(subset=need).copy();assert len(d)==source_n
    print(f"Fitting Model {number} {specification}: {len(d):,} threads.",flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result=smf.logit(formula,data=d).fit(cov_type="cluster",cov_kwds={"groups":d.DMRN},disp=False,maxiter=100)
    assert result.mle_retvals["converged"]
    covariance=result.cov_params()
    assert np.isfinite(covariance.to_numpy()).all() and np.isfinite(result.params.to_numpy()).all()
    assert np.allclose(covariance.to_numpy(),covariance.to_numpy().T,atol=1e-12)
    assert np.linalg.eigvalsh(covariance.to_numpy()).min()>-1e-10
    assert np.allclose(covariance.to_numpy(),cov_cluster(result,d.DMRN.to_numpy(),use_correction=True),rtol=1e-8,atol=1e-10)
    z=norm.ppf(.975)
    coefficients=pd.DataFrame({"term":result.params.index,"beta":result.params.to_numpy(),"cluster_robust_se":result.bse.to_numpy(),"OR":np.exp(result.params.to_numpy()),"CI_low":np.exp(result.params.to_numpy()-z*result.bse.to_numpy()),"CI_high":np.exp(result.params.to_numpy()+z*result.bse.to_numpy()),"P":result.pvalues.to_numpy()})
    coefficients["OR_95_CI"]=coefficients.apply(lambda r:f"{r.OR:.2f} ({r.CI_low:.2f}–{r.CI_high:.2f})",axis=1)
    stem=f"model_{number}_{specification}"
    coefficients.to_csv(run/(stem+"_coefficients.csv"),index=False)
    covariance.to_csv(run/(stem+"_patient_clustered_covariance.csv"))
    terms=[("Physician/APP messages","log_physapp_msgs_past_30d"),("Other HCW messages","log_other_human_msgs_past_30d"),("System messages","log_system_msgs_past_30d"),("Physician/APP × other HCW","log_physapp_msgs_past_30d:log_other_human_msgs_past_30d")]
    rows=[]
    for label,term in terms:
        row=coefficients.set_index("term").loc[term].to_dict()
        rows.append({"model":number,"specification":specification,"exposure":label,"term":term,**row})
    diagnostic={"model":number,"specification":specification,"n_threads":len(d),"n_patients":d.DMRN.nunique(),"replies_7d":int(d.patient_replied_7d.sum()),"reply_percent":100*d.patient_replied_7d.mean(),"converged":True,"iterations":int(result.mle_retvals["iterations"]),"aic":result.aic,"formula":formula,"patient_clustered_covariance_independently_verified":True,"warnings":[str(w.message) for w in caught]}
    print("Converged.",flush=True)
    del result;gc.collect()
    return rows,diagnostic

def run_analysis():
    from .encounters import formula
    run=new_run("message_type_sensitivity",[PRIMARY,MESSAGES])
    a,corpus=load_analysis(); descriptions(a,corpus,run)
    rows=[];diagnostics=[]
    samples={2:a.loc[a.thread_initiator_group.eq("PhysAPP")].copy(),
             3:a.loc[a.thread_initiator_group.eq("OtherHuman")].copy()}
    for number,d in samples.items():
        r,z=fit(d,number,"original",formula(number,30),run);rows+=r;diagnostics.append(z)
        grouped=grouped_type(d.copy(),number,run)
        r,z=fit(grouped,number,"type_adjusted",
            formula(number,30)+' + C(model_message_type, Treatment(reference="User Message"))',run)
        rows+=r;diagnostics.append(z)
    base=samples[3]
    excluded=base.initiator_structured_role.isin(["Administration","Unknown"])
    retained=base.loc[~excluded].copy()
    assert int(excluded.sum())==56282 and len(retained)==318230
    r,z=fit(retained,3,"exclude_administration_unknown",formula(3,30),run)
    rows+=r;diagnostics.append(z)
    pd.DataFrame(rows).to_csv(run/"Table_S11_sensitivity_exposure_estimates.csv",index=False)
    save(run/"model_diagnostics.json",diagnostics)
    pd.DataFrame(diagnostics).drop(columns=["formula","warnings"]).to_csv(run/"model_samples.csv",index=False)
    return finish(run)
