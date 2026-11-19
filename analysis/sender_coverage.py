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

import ast
from .common import save
def run_analysis():
    messages_path=MESSAGES
    analysis_path=PRIMARY
    master_path=latest_run("core_message_preprocessing_",["provider_master_clean.parquet"])/"provider_master_clean.parquet"
    source01=ROOT/"notebooks/01_build_core_message_preprocessing_clean.ipynb"
    run=new_run("sender_mapping_coverage",[messages_path,analysis_path,master_path,source01])
    mapping=None
    nb=json.loads(source01.read_text(encoding="utf-8-sig"))
    for cell in nb["cells"]:
        if cell["cell_type"]!="code":continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="role_mapping" for x in node.targets):
                mapping=ast.literal_eval(node.value)
    assert mapping is not None
    master=pd.read_parquet(master_path,columns=["USER_ID","PROV_TYPE","Type_BUCKET"])
    master["USER_ID"]=master.USER_ID.astype("string").str.strip()
    assert master.USER_ID.notna().all() and master.USER_ID.is_unique
    expected=master.PROV_TYPE.map(mapping).fillna("Unknown")
    assert expected.equals(master.Type_BUCKET.astype(expected.dtype))
    data=pd.read_parquet(messages_path,columns=["MESSAGE_ID","FROM_USER_ID","SENDER_CATEGORY","Sender_all","is_system_generated"])
    assert len(data)==3237639 and data.MESSAGE_ID.is_unique
    data["FROM_USER_ID"]=data.FROM_USER_ID.astype("string").str.strip()
    data["MESSAGE_ID"]=data.MESSAGE_ID.astype("string").str.strip()
    human=~data.Sender_all.isin(["Patient","System"]) & data.is_system_generated.fillna(0).eq(0)
    assert not data.loc[human,"FROM_USER_ID"].isna().any()
    assert not data.loc[human,"FROM_USER_ID"].eq("Patient").any()
    sender_ids=data.loc[human,["FROM_USER_ID"]].drop_duplicates()
    id_lookup=sender_ids.merge(master,left_on="FROM_USER_ID",right_on="USER_ID",how="left",validate="one_to_one",indicator=True)
    total_ids=len(id_lookup)
    linked_ids=int(id_lookup["_merge"].eq("both").sum())
    assert total_ids==linked_ids==7542
    nonmissing_type=int(id_lookup.PROV_TYPE.notna().sum())
    recognized_initial_bucket=int((id_lookup.Type_BUCKET.notna() & id_lookup.Type_BUCKET.ne("Unknown")).sum())
    coverage=[
        {"scope":"Full corpus HCW senders","measure":"Distinct sender IDs linked to lookup records","numerator":linked_ids,"denominator":total_ids,"percent":100*linked_ids/total_ids},
        {"scope":"Full corpus HCW senders","measure":"Nonmissing provider-type field (presence only)","numerator":nonmissing_type,"denominator":total_ids,"percent":100*nonmissing_type/total_ids},
        {"scope":"Full corpus HCW senders","measure":"Non-Unknown initial broad role bucket, before workflow refinements","numerator":recognized_initial_bucket,"denominator":total_ids,"percent":100*recognized_initial_bucket/total_ids},
    ]
    # APP/physician membership agrees between frozen provider lookup and final
    # message classification; refinements can affect granular Other HCW labels.
    joined=data.loc[human,["MESSAGE_ID","FROM_USER_ID","Sender_all"]].merge(master,left_on="FROM_USER_ID",right_on="USER_ID",how="left",validate="many_to_one")
    assert joined.Sender_all.isin(["Physician","APP"]).equals(joined.Type_BUCKET.isin(["Physician","APP"]))
    app_types=joined.loc[joined.Sender_all.eq("APP")].groupby("PROV_TYPE",dropna=False).agg(unique_sender_ids=("FROM_USER_ID","nunique"),messages=("MESSAGE_ID","size")).reset_index()
    unknown_types=joined.loc[joined.Type_BUCKET.eq("Unknown")].groupby("PROV_TYPE",dropna=False).agg(unique_sender_ids=("FROM_USER_ID","nunique"),messages=("MESSAGE_ID","size")).reset_index()
    analysis=pd.read_parquet(analysis_path,columns=["thread_id","thread_first_message_id","thread_initiator_group","patient_replied_7d"])
    assert len(analysis)==458525 and analysis.thread_id.is_unique
    analysis["thread_first_message_id"]=analysis.thread_first_message_id.astype("string").str.strip()
    first=analysis.merge(data,left_on="thread_first_message_id",right_on="MESSAGE_ID",how="left",validate="many_to_one",indicator=True)
    assert first["_merge"].eq("both").all()
    derived_group=np.where(first.Sender_all.isin(["Physician","APP"]),"PhysAPP","OtherHuman")
    assert (derived_group==first.thread_initiator_group).all()
    initiator_ids=first[["FROM_USER_ID"]].drop_duplicates().merge(master,left_on="FROM_USER_ID",right_on="USER_ID",how="left",validate="one_to_one",indicator=True)
    assert len(initiator_ids)==6296 and initiator_ids["_merge"].eq("both").all()
    coverage.append({"scope":"Primary analytic HCW-thread initiators","measure":"Distinct sender IDs linked to lookup records","numerator":6296,"denominator":6296,"percent":100.})
    other=first.loc[first.thread_initiator_group.eq("OtherHuman")].copy()
    assert len(other)==374512
    other["role"]=other.SENDER_CATEGORY.astype("string").str.strip().replace("",pd.NA).fillna("Unknown")
    roles=other.groupby("role").agg(threads=("thread_id","size"),replies=("patient_replied_7d","sum")).reset_index()
    roles["percent_of_other_hcw_threads"]=100*roles.threads/len(other)
    roles["reply_percent"]=100*roles.replies/roles.threads
    expected_counts={"Support Staff":(144598,25884),"Nurse":(104158,29278),"Administration":(40511,1223),
                     "Allied Health":(29966,8720),"Behavioral Health":(24088,5279),"Unknown":(15771,2690),
                     "Therapist":(8715,3647),"Tech Specialist":(4879,821),"Student":(1826,444)}
    assert {row.role:(row.threads,row.replies) for row in roles.itertuples(index=False)}==expected_counts
    assert other.patient_replied_7d.sum()==77986
    unknown=other.loc[other.role.eq("Unknown")]
    assert len(unknown)==15771 and unknown.thread_initiator_group.eq("OtherHuman").all()
    # Some initial Unknown roles receive an Administration label during the
    # original workflow refinement. Both remain OtherHuman; S9 uses initial roles.
    unknown_final=unknown.Sender_all.value_counts().rename_axis("final_role").reset_index(name="threads")
    assert set(unknown_final.final_role)<=set(["Unknown","Administration"])
    assert dict(zip(unknown_final.final_role,unknown_final.threads))=={"Unknown":14213,"Administration":1558}

    pd.DataFrame(coverage).to_csv(run/"sender_id_linkage_coverage.csv",index=False)
    roles.sort_values("threads",ascending=False).to_csv(run/"Table_S9_sender_roles.csv",index=False)
    app_types.to_csv(run/"observed_app_provider_types.csv",index=False)
    unknown_types.to_csv(run/"initial_unknown_bucket_provider_types.csv",index=False)
    unknown_final.to_csv(run/"unknown_initial_vs_final_role_counts.csv",index=False)
    pd.DataFrame([{"provider_type_label":key,"role_bucket":value} for key,value in mapping.items()]).to_csv(run/"provider_role_dictionary.csv",index=False)
    return finish(run)
