from __future__ import annotations
import os
for _key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ.setdefault(_key,"4")
from pathlib import Path
from datetime import datetime,timezone
import gc,hashlib,json,sys,warnings,time,shutil
import numpy as np
import pandas as pd
import patsy
import scipy,statsmodels
import statsmodels.formula.api as smf
from scipy.stats import norm
from scipy.special import expit,xlogy,logit
from statsmodels.stats.sandwich_covariance import cov_cluster
from .common import ROOT,OUTPUT_ROOT,WORKING,PRIMARY,MESSAGES,ENCOUNTERS,new_run,finish,require,latest_run

LEGACY=WORKING
THREAD_RUN=None
START=pd.Timestamp("2021-03-31")
END=pd.Timestamp("2023-11-30 23:59:59.999999999")
EXPOSURES=["provider_msgs_past_30d","physapp_msgs_past_30d","other_human_msgs_past_30d","system_msgs_past_30d","patient_initiated_threads_past_30d"]
COVARIATES=["Age_at_Message","Age65","Gender","Race","Ethnicity","Charlson_group"]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for part in iter(lambda:f.read(8*1024*1024),b""):h.update(part)
    return h.hexdigest()

def write_json(p,value):
    Path(p).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")

def normalize(s,label):
    return s.astype("string").str.strip().str.lower().replace("",pd.NA).fillna(label)

def mode_per_patient(d,col):
    """Match notebook 05 mode_or_first: nonmissing mode, smallest value on ties."""
    counts=d.dropna(subset=[col]).groupby(["DMRN",col],observed=True,sort=False).size().reset_index(name="_n")
    return counts.sort_values(["DMRN","_n",col],ascending=[True,False,True]).drop_duplicates("DMRN").set_index("DMRN")[col]

def load_metadata():
    cols=["MESSAGE_ID","DMRN","CREATED_TIME","MESG_TYPE","Prov_DEPT_For_Appt","SUBJECT_CLEAN","Sender_all","is_system_generated","thread_id"]
    m=pd.read_parquet(LEGACY/"messages_with_threads.parquet",columns=cols)
    assert len(m)==3237639 and m.MESSAGE_ID.is_unique
    assert m[["DMRN","CREATED_TIME","Sender_all","thread_id"]].notna().all().all()
    m["DMRN"]=m.DMRN.astype(str).str.strip()
    m["_id"]=pd.to_numeric(m.MESSAGE_ID,errors="raise")
    m["subject_key"]=normalize(m.SUBJECT_CLEAN,"unknown_subject").astype("category")
    m["department_key"]=normalize(m.Prov_DEPT_For_Appt,"unknown_department").astype("category")
    m["type_key"]=normalize(m.MESG_TYPE,"unknown_message_type").astype("category")
    m["subject_missing"]=m.SUBJECT_CLEAN.isna()|m.SUBJECT_CLEAN.astype("string").str.strip().eq("")
    m["sender_group"]=np.select([m.Sender_all.eq("Patient"),m.Sender_all.eq("System")|m.is_system_generated.eq(1),m.Sender_all.isin(["Physician","APP"])],["Patient","System","PhysAPP"],default="OtherHuman")
    for c in ["DMRN","MESG_TYPE","Prov_DEPT_For_Appt","SUBJECT_CLEAN","Sender_all","sender_group"]:m[c]=m[c].astype("category")
    return m

def load_covariates(canonical):
    print("Reconstructing the original patient-level mode covariates.",flush=True)
    age=pd.read_parquet(WORKING/"message_demographics_clean.parquet",columns=["DMRN","Age_at_Message","AgeGroup65"])
    age["DMRN"]=age.DMRN.astype(str).str.strip()
    age["Age65"]=age.AgeGroup65.map({0:"<65",1:"≥65"})
    cov=pd.DataFrame({c:mode_per_patient(age,c) for c in ["Age_at_Message","Age65"]})
    del age;gc.collect()
    cci=pd.read_parquet(WORKING/"message_level_charlson_cci.parquet",columns=["DMRN","Charlson_group"])
    cci["DMRN"]=cci.DMRN.astype(str).str.strip()
    cov=cov.join(mode_per_patient(cci,"Charlson_group"));del cci;gc.collect()
    demo=pd.read_parquet(WORKING/"patient_demographics_clean.parquet",columns=["DMRN","Gender","Race_Grouped","Ethnicity_Grouped"])
    demo["DMRN"]=demo.DMRN.astype(str).str.strip()
    demo=demo.drop_duplicates("DMRN").set_index("DMRN").rename(columns={"Race_Grouped":"Race","Ethnicity_Grouped":"Ethnicity"})
    cov=cov.join(demo)
    original=canonical[["DMRN"]+COVARIATES].drop_duplicates()
    assert original.DMRN.is_unique
    test=original.merge(cov.reset_index(),on="DMRN",suffixes=("_saved","_rebuilt"),validate="one_to_one")
    assert len(test)==canonical.DMRN.nunique()
    checks={c:int(test[c+"_saved"].astype(str).ne(test[c+"_rebuilt"].astype(str)).sum()) for c in COVARIATES}
    assert not any(checks.values()),checks
    return cov,checks

def ordered_first(m,thread_col):
    return m.sort_values([thread_col,"CREATED_TIME","_id"],kind="stable").drop_duplicates(thread_col)

def cross_key_audit(m,canonical,run):
    print("Auditing same-patient, same-subject messages outside the recorded thread.",flush=True)
    first=m.set_index("MESSAGE_ID")
    anchors=canonical[["thread_first_message_id","thread_id","DMRN","thread_first_time","thread_initiator_group","patient_replied_7d","patient_replied_ever"]].copy()
    linked=anchors.join(first[["subject_key","department_key","type_key","subject_missing"]],on="thread_first_message_id",validate="one_to_one")
    assert linked.subject_key.notna().all()
    # Patients without a reply within 7 days cannot have an in-thread patient
    # message during that window. Thus the first same-subject patient message,
    # if found, must belong to another thread; this is checked explicitly.
    left=linked.loc[linked.patient_replied_7d.eq(0)].copy()
    right=m.loc[m.sender_group.eq("Patient"),["DMRN","CREATED_TIME","thread_id","subject_key","department_key","type_key"]].rename(columns={"CREATED_TIME":"candidate_time","thread_id":"candidate_thread_id","department_key":"candidate_department_key","type_key":"candidate_type_key"})
    for c in ["DMRN","subject_key"]:
        left[c]=left[c].astype(str);right[c]=right[c].astype(str)
    merged=pd.merge_asof(left.sort_values("thread_first_time",kind="stable"),right.sort_values("candidate_time",kind="stable"),left_on="thread_first_time",right_on="candidate_time",by=["DMRN","subject_key"],direction="forward",tolerance=pd.Timedelta(days=7),allow_exact_matches=True)
    merged["candidate_found"]=merged.candidate_time.notna()
    assert merged.loc[merged.candidate_found,"candidate_thread_id"].astype(str).str.replace(r"\.0$","",regex=True).ne(merged.loc[merged.candidate_found,"thread_id"].astype(str)).all()
    # Compare normalized keys for the earliest qualifying candidate only.
    hit=merged.candidate_found
    dept=merged.department_key.astype("string").eq(merged.candidate_department_key.astype("string")).fillna(False)
    typ=merged.type_key.astype("string").eq(merged.candidate_type_key.astype("string")).fillna(False)
    assert not (hit&dept&typ).any()
    merged["candidate_key_difference"]=np.select([hit&dept&~typ,hit&~dept&typ,hit&~dept&~typ],["message type only","department only","department and message type"],default="no candidate")
    rows=[]
    for label,mask in [("No observed first reply during available follow-up",merged.patient_replied_ever.eq(0)),("No first reply within 7 days",pd.Series(True,index=merged.index))]:
        for group in ["All HCW","PhysAPP","OtherHuman"]:
            d=merged.loc[mask & (True if group=="All HCW" else merged.thread_initiator_group.eq(group))]
            n=len(d);k=int(d.candidate_found.sum())
            rows.append({"denominator_definition":label,"initiator_group":group,"threads":n,"with_same_subject_patient_message_in_other_thread_within_7d":k,"percent":100*k/n,
                         "earliest_candidate_type_only":int(d.candidate_key_difference.eq("message type only").sum()),"earliest_candidate_department_only":int(d.candidate_key_difference.eq("department only").sum()),"earliest_candidate_both":int(d.candidate_key_difference.eq("department and message type").sum()),
                         "missing_or_empty_subject_threads":int(d.subject_missing.sum()),"same_timestamp_candidate_threads":int((d.candidate_time.eq(d.thread_first_time)&d.candidate_found).sum())})
    result=pd.DataFrame(rows);result.to_csv(run/"cross_key_reply_audit.csv",index=False)
    # Codes of already-linked replies agree because message type is a key,
    # not because these are independently adjudicated conversation matches.
    linked_reply=canonical.loc[canonical.patient_replied_ever.eq(1),["thread_id","first_patient_reply_time"]].copy()
    reply_types=m.loc[m.sender_group.eq("Patient"),["thread_id","CREATED_TIME","type_key"]].drop_duplicates(["thread_id","CREATED_TIME"])
    linked_reply=linked_reply.merge(reply_types,left_on=["thread_id","first_patient_reply_time"],right_on=["thread_id","CREATED_TIME"],how="left",validate="one_to_one")
    linked_reply=linked_reply.merge(linked[["thread_id","type_key"]],on="thread_id",suffixes=("_reply","_initiator"),validate="one_to_one")
    assert linked_reply.type_key_reply.notna().all()
    agree=linked_reply.type_key_reply.astype(str).eq(linked_reply.type_key_initiator.astype(str))
    stats={"linked_first_replies":len(linked_reply),"same_normalized_message_type":int(agree.sum()),"percent":100*agree.mean(),"interpretation":"Agreement is required by the reconstruction key; it is not an independent validation of conversation linkage.","native_conversation_identifier_available_in_analysis_extract":False,"candidate_interpretation":"Same patient and normalized subject within 7 days; different thread; candidate messages are not confirmed replies and outcomes are not recoded."}
    assert agree.all();write_json(run/"message_type_linkage_audit.json",stats)
    print(result.to_string(index=False),flush=True)
    return result,stats

def prepare_events(m):
    events=m.sort_values(["DMRN","CREATED_TIME"],kind="stable").reset_index(drop=True)
    times=events.CREATED_TIME.to_numpy(dtype="datetime64[ns]")
    groups=events.groupby("DMRN",observed=True,sort=False).indices
    cum={
        "provider_msgs_past_30d":np.r_[0,events.sender_group.isin(["PhysAPP","OtherHuman"]).to_numpy().cumsum()],
        "physapp_msgs_past_30d":np.r_[0,events.sender_group.eq("PhysAPP").to_numpy().cumsum()],
        "other_human_msgs_past_30d":np.r_[0,events.sender_group.eq("OtherHuman").to_numpy().cumsum()],
        "system_msgs_past_30d":np.r_[0,events.sender_group.eq("System").to_numpy().cumsum()]}
    enc=pd.read_parquet(LEGACY/"encounters_unified_clean.parquet",columns=["DMRN","ENCOUNTER_START"])
    enc=enc.dropna();enc["DMRN"]=enc.DMRN.astype(str).str.strip()
    enc=enc.sort_values(["DMRN","ENCOUNTER_START"],kind="stable")
    encounters={pid:g.ENCOUNTER_START.to_numpy(dtype="datetime64[ns]") for pid,g in enc.groupby("DMRN",sort=False)}
    return times,groups,cum,encounters

def make_analysis(m,cutoff,cov,events,canonical,run):
    print(f"Rebuilding {cutoff}-day gap analysis frame.",flush=True)
    saved=pd.read_parquet(THREAD_RUN/f"messages_threaded_{cutoff}d.parquet",columns=["MESSAGE_ID","thread_id"])
    assert saved.MESSAGE_ID.is_unique
    # IDs differ between the historical sensitivity run and the canonical
    # handoff, so assign by MESSAGE_ID and validate partitions at 30 days.
    assignment=m[["MESSAGE_ID","thread_id"]].merge(saved,on="MESSAGE_ID",how="outer",validate="one_to_one",suffixes=("_primary","_sensitivity"),indicator=True)
    assert assignment._merge.eq("both").all()
    assignment=assignment.set_index("MESSAGE_ID").loc[m.MESSAGE_ID.to_numpy()].reset_index()
    if cutoff==30:
        partition=assignment[["thread_id_primary","thread_id_sensitivity"]].drop_duplicates()
        assert partition.thread_id_primary.is_unique and partition.thread_id_sensitivity.is_unique
    mt=m.copy(deep=False).assign(sensitivity_thread_id=assignment.thread_id_sensitivity.to_numpy())
    # The outer merge sorts keys; ensure assignment is restored to m order.
    assert np.array_equal(assignment.MESSAGE_ID.to_numpy(),m.MESSAGE_ID.to_numpy()),"Message assignment order must match current metadata."
    first=ordered_first(mt,"sensitivity_thread_id")
    assert len(first)=={15:1475330,30:1399147,60:1324689}[cutoff]
    patient_starts=first.loc[first.sender_group.eq("Patient"),["DMRN","CREATED_TIME"]].sort_values(["DMRN","CREATED_TIME"],kind="stable")
    patient_times={str(pid):g.CREATED_TIME.to_numpy(dtype="datetime64[ns]") for pid,g in patient_starts.groupby("DMRN",observed=True,sort=False)}
    a=first.loc[first.sender_group.isin(["PhysAPP","OtherHuman"]),["sensitivity_thread_id","MESSAGE_ID","DMRN","CREATED_TIME","Sender_all","sender_group"]].rename(columns={"sensitivity_thread_id":"thread_id","MESSAGE_ID":"thread_first_message_id","CREATED_TIME":"thread_first_time","Sender_all":"thread_initiator_sender","sender_group":"thread_initiator_group"}).copy()
    a["DMRN"]=a.DMRN.astype(str)
    a=a.merge(cov.reset_index(),on="DMRN",how="left",validate="many_to_one")
    a=a.loc[a.thread_first_time.between(START,END)&a.Age_at_Message.notna()&a.Age65.notna()].reset_index(drop=True)
    # Recompute first patient replies using the unchanged notebook 05 rule.
    patient=mt.loc[mt.sender_group.eq("Patient"),["sensitivity_thread_id","CREATED_TIME"]].groupby("sensitivity_thread_id",sort=False).CREATED_TIME.min()
    a["first_patient_reply_time"]=a.thread_id.map(patient)
    assert (a.first_patient_reply_time.isna()|a.first_patient_reply_time.ge(a.thread_first_time)).all()
    for d in [7,14,30]:a[f"patient_replied_{d}d"]=(a.first_patient_reply_time.notna()&a.first_patient_reply_time.le(a.thread_first_time+pd.Timedelta(days=d))).astype(int)
    a["patient_replied_ever"]=a.first_patient_reply_time.notna().astype(int)
    times,groups,cumulative,encounters=events
    counts={c:np.zeros(len(a),dtype=np.int32) for c in EXPOSURES}
    encounter_counts=np.zeros(len(a),dtype=np.int32)
    for pid,ix in a.groupby("DMRN",sort=False).indices.items():
        t=a.thread_first_time.to_numpy(dtype="datetime64[ns]")[ix]
        j=groups[pid];start=j[0];tt=times[j]
        left=np.searchsorted(tt,t-np.timedelta64(30,"D"),side="left")+start
        right=np.searchsorted(tt,t,side="right")+start
        for col,arr in cumulative.items():counts[col][ix]=arr[right]-arr[left]
        pt=patient_times.get(pid,np.array([],dtype="datetime64[ns]"))
        counts["patient_initiated_threads_past_30d"][ix]=np.searchsorted(pt,t,side="right")-np.searchsorted(pt,t-np.timedelta64(30,"D"),side="left")
        et=encounters.get(pid,np.array([],dtype="datetime64[ns]"))
        encounter_counts[ix]=np.searchsorted(et,t,side="left")-np.searchsorted(et,t-np.timedelta64(7,"D"),side="left")
    for c,arr in counts.items():a[c]=arr;a["log_"+c]=np.log1p(arr.astype(float))
    a["encounters_past_7d"]=encounter_counts
    a["log_encounters_past_7d"]=np.log1p(encounter_counts.astype(float))
    assert a.provider_msgs_past_30d.eq(a.physapp_msgs_past_30d+a.other_human_msgs_past_30d).all()
    # Match notebook 06 preparation and reference categories exactly.
    a["Age65"]=pd.Categorical(a.Age65,categories=["<65","≥65"])
    a["Gender"]=a.Gender.astype("category")
    a["Ethnicity"]=pd.Categorical(a.Ethnicity,categories=["Not Hispanic/Latino","Hispanic/Latino","Unknown/Other"])
    a["Charlson_group"]=a.Charlson_group.astype("category")
    a["Race"]=pd.Categorical(np.select([a.Race.eq("White"),a.Race.eq("Black/African American"),a.Race.eq("Asian")],["White","Black/African American","Asian"],default="Other/Multiple/Unknown"),categories=["White","Black/African American","Asian","Other/Multiple/Unknown"])
    a["month_year"]=a.thread_first_time.dt.to_period("M").astype(str)
    if cutoff==30:
        assert len(a)==458525 and a.thread_initiator_group.eq("PhysAPP").sum()==84013
        compare_columns=EXPOSURES+["patient_replied_7d","patient_replied_14d","patient_replied_30d","patient_replied_ever","encounters_past_7d","Age_at_Message"]
        z=a[["thread_first_message_id"]+compare_columns].merge(canonical[["thread_first_message_id"]+compare_columns],on="thread_first_message_id",how="outer",suffixes=("_rebuilt","_saved"),validate="one_to_one",indicator=True)
        assert z._merge.eq("both").all()
        checks={c:int(z[c+"_rebuilt"].ne(z[c+"_saved"]).sum()) for c in compare_columns}
        assert not any(checks.values()),checks
        write_json(run/"primary_30d_dataset_reproduction.json",{"verified":True,"n_threads":len(a),"n_patients":a.DMRN.nunique(),"different_thread_identifier_numbering_only":True,"mismatches":checks})
    return a

def fit_models(a,cutoff,run):
    base='log_encounters_past_7d + Age_at_Message + C(Gender) + C(Race, Treatment(reference="White")) + Ethnicity + C(Charlson_group) + C(month_year)'
    formulas={1:'patient_replied_7d ~ log_provider_msgs_past_30d + log_system_msgs_past_30d + log_patient_initiated_threads_past_30d + log_provider_msgs_past_30d:log_system_msgs_past_30d + '+base,
              2:'patient_replied_7d ~ log_physapp_msgs_past_30d + log_other_human_msgs_past_30d + log_system_msgs_past_30d + log_patient_initiated_threads_past_30d + log_physapp_msgs_past_30d:log_other_human_msgs_past_30d + '+base}
    rows=[];diagnostics=[];z=norm.ppf(.975)
    for number,formula in formulas.items():
        d=a if number==1 else a.loc[a.thread_initiator_group.eq("PhysAPP")].copy()
        needed=["DMRN","patient_replied_7d","Age_at_Message","Gender","Race","Ethnicity","Charlson_group","month_year","log_encounters_past_7d"]+["log_"+c for c in EXPOSURES]
        d=d.dropna(subset=needed).copy()
        print(f"Fitting gap={cutoff} days, Model {number}, N={len(d):,}.",flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result=smf.logit(formula,data=d).fit(cov_type="cluster",cov_kwds={"groups":d.DMRN},disp=False,maxiter=100)
        assert result.mle_retvals["converged"]
        covariance=result.cov_params();assert np.isfinite(covariance.to_numpy()).all()
        coeff=pd.DataFrame({"term":result.params.index,"beta":result.params.to_numpy(),"cluster_robust_se":result.bse.to_numpy(),"OR":np.exp(result.params.to_numpy()),"CI_low":np.exp(result.params.to_numpy()-z*result.bse.to_numpy()),"CI_high":np.exp(result.params.to_numpy()+z*result.bse.to_numpy()),"P":result.pvalues.to_numpy()})
        coeff.to_csv(run/f"gap_{cutoff}d_model_{number}_coefficients.csv",index=False)
        covariance.to_csv(run/f"gap_{cutoff}d_model_{number}_patient_clustered_covariance.csv")
        wanted={1:[("HCW messages","log_provider_msgs_past_30d"),("System messages","log_system_msgs_past_30d"),("HCW × system","log_provider_msgs_past_30d:log_system_msgs_past_30d")],2:[("Physician/APP messages","log_physapp_msgs_past_30d"),("Other HCW messages","log_other_human_msgs_past_30d"),("System messages","log_system_msgs_past_30d"),("Physician/APP × other HCW","log_physapp_msgs_past_30d:log_other_human_msgs_past_30d")]}
        for label,term in wanted[number]:
            r=coeff.set_index("term").loc[term].to_dict()
            rows.append({"gap_days":cutoff,"model":number,"exposure":label,"term":term,**r,"OR_95_CI":f"{r['OR']:.2f} ({r['CI_low']:.2f}–{r['CI_high']:.2f})"})
        diagnostics.append({"gap_days":cutoff,"model":number,"n_threads":len(d),"n_patients":d.DMRN.nunique(),"n_replies_7d":int(d.patient_replied_7d.sum()),"reply_percent":100*d.patient_replied_7d.mean(),"converged":True,"iterations":int(result.mle_retvals["iterations"]),"aic":result.aic,"formula":formula,"warnings":[str(w.message) for w in caught]})
        print("Converged.",flush=True)
        del result,d;gc.collect()
    return rows,diagnostics

def run_analysis():
    global THREAD_RUN
    THREAD_RUN=latest_run("threading_diagnostics_sensitivity_",[
        "messages_threaded_15d.parquet","messages_threaded_30d.parquet",
        "messages_threaded_60d.parquet","threading_sensitivity_summary.csv"])
    run=new_run("thread_reconstruction_sensitivity",[PRIMARY,MESSAGES])
    m=load_metadata();canonical=pd.read_parquet(PRIMARY)
    canonical["thread_id"]=pd.to_numeric(canonical.thread_id,errors="raise")
    cross_key_audit(m,canonical,run)
    cov,checks=load_covariates(canonical);events=prepare_events(m)
    rows=[];diagnostics=[]
    for days in [30,15,60]:
        a=make_analysis(m,days,cov,events,canonical,run)
        r,z=fit_models(a,days,run);rows+=r;diagnostics+=z
        del a;gc.collect()
    pd.DataFrame(rows).to_csv(run/"gap_sensitivity_exposure_ORs.csv",index=False)
    pd.DataFrame(rows).pivot(index=["model","exposure"],columns="gap_days",values="OR_95_CI").reset_index().to_csv(run/"Table_S4_panel_B_model_ORs.csv",index=False)
    pd.DataFrame(diagnostics).drop(columns=["formula","warnings"]).to_csv(run/"gap_sensitivity_model_samples.csv",index=False)
    write_json(run/"model_diagnostics.json",diagnostics)
    write_json(run/"covariate_reconstruction_checks.json",checks)
    # Panel A is the original thread-characteristic summary from notebook 02.
    shutil.copy2(THREAD_RUN/"threading_sensitivity_summary.csv",run/"Table_S4_panel_A_thread_characteristics.csv")
    return finish(run)
