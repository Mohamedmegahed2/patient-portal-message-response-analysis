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

from decimal import Decimal,ROUND_HALF_UP
def one_decimal(value):
    return str(Decimal(str(value)).quantize(Decimal("0.1"),rounding=ROUND_HALF_UP))
def run_analysis():
    source=PRIMARY
    run=new_run("reply_time_distribution",[source])
    data=pd.read_parquet(source,columns=["thread_id","thread_first_time","first_patient_reply_time","reply_delay_days",
                                       "patient_replied_7d","patient_replied_14d","patient_replied_30d","patient_replied_ever"])
    assert len(data)==458525 and data.thread_id.is_unique
    anchors=pd.to_datetime(data.thread_first_time,errors="raise")
    replies=pd.to_datetime(data.first_patient_reply_time,errors="raise")
    hours=(replies-anchors).dt.total_seconds()/3600
    assert not hours.lt(0).any()
    observed=hours.notna() & hours.ge(0)
    n_observed=int(observed.sum())
    n30=int((observed & hours.le(720)).sum())
    n_after30=int((observed & hours.gt(720)).sum())
    assert n_observed==128436 and n30==128192 and n_after30==244
    assert data.patient_replied_ever.eq(1).equals(observed)
    assert np.allclose(data.reply_delay_days.to_numpy(dtype=float),hours.to_numpy()/24,equal_nan=True,atol=1e-12)
    rows=[]
    for i,days in enumerate([7,14,30]):
        within=observed & hours.le(24*days)
        count=int(within.sum())
        flag=data[f"patient_replied_{days}d"].eq(1)
        assert flag.equals(within)
        pct_all=100*count/n_observed
        rows.append({"days":days,"first_reply_count":count,"all_observed_replies_denominator":n_observed,
                     "percent_all_observed_replies":pct_all,"within30_replies_denominator":n30,
                     "percent_among_within30_replies":100*count/n30,"all_threads_denominator":len(data),
                     "percent_all_threads":100*count/len(data),"binary_flag_count":int(flag.sum()),"timestamp_flag_disagreements":0})
    quantiles=hours.loc[observed].quantile([.25,.5,.75])
    assert np.allclose(quantiles.to_numpy(),[.6,4.016666666666667,49.15])
    rounded={"q25":one_decimal(quantiles.loc[.25]),"median":one_decimal(quantiles.loc[.5]),"q75":one_decimal(quantiles.loc[.75])}
    assert rounded=={"q25":"0.6","median":"4.0","q75":"49.2"}
    pd.DataFrame(rows).to_csv(run/"reply_window_denominators.csv",index=False)
    pd.DataFrame([{"percentile":q,"reply_delay_hours":v,"reported_one_decimal":one_decimal(v)} for q,v in quantiles.items()]).to_csv(run/"observed_reply_time_quantiles.csv",index=False)
    return finish(run)
