"""Aggregate-only handoffs from the existing primary and subgroup fits."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import save,sha
from .predictions import design_vector

def export_model(obj,profile,path,settings):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    r=obj["model"]
    assert r.mle_retvals["converged"]
    cov=r.cov_params().loc[r.params.index,r.params.index]
    pd.DataFrame({"term":r.params.index,"beta":r.params.to_numpy(),
        "cluster_robust_se":r.bse.to_numpy()}).to_csv(path/"coefficients.csv",index=False)
    cov.to_csv(path/"patient_clustered_covariance.csv")
    save(path/"profile.json",profile)
    settings.update({"formula":r.model.formula,"n_threads":int(r.nobs),
        "converged":True,"covariance":"patient clustered","profile_type":"conditional representative profile"})
    save(path/"settings.json",settings)
    # Verify reconstruction of categorical and interaction columns against Patsy.
    from scipy.special import expit
    xlog=settings["focal_log_column"]
    for count in [settings["focal_minimum"],5.,settings["focal_p99"]]:
        row={**profile,xlog:float(np.log1p(count))}
        reconstructed=float(expit(design_vector(r.params.index.tolist(),profile,{xlog:row[xlog]})@r.params.to_numpy()))
        original=float(r.predict(pd.DataFrame([row])).iloc[0])
        assert np.isclose(reconstructed,original,atol=1e-12,rtol=1e-12)
    save(path/"manifest.json",{p.name:sha(p) for p in path.iterdir() if p.is_file() and p.name!="manifest.json"})

def export_primary(panels,run):
    root=Path(run)/"aggregate_models"
    for letter,p in panels.items():
        d=p["df"];x=p["x_cnt_col"];m=p["m_cnt_col"]
        profile=p["profile"]
        settings={"model":dict(A=1,B=2,C=3)[letter],"focal_column":x,
            "focal_log_column":p["x_log_name"],"moderator_column":m,
            "moderator_log_column":p["m_log_name"],
            "focal_minimum":float(d[x].min()),"focal_p99":float(d[x].quantile(.99)),
            "focal_percentiles":{str(k):float(d[x].quantile(k/100)) for k in [50,90]},
            "moderator_percentiles":{str(k):float(d[m].quantile(k/100)) for k in [50,90,99]},
            "month_year":str(profile["month_year"])}
        export_model(p["obj"],profile,root/letter,settings)
    save(root/"COMPLETE.json",{"complete":True,"models":["A","B","C"]})
    return root

def export_subgroups(registry,profile_builder,run):
    root=Path(run)/"aggregate_models"
    for name in ["Sex","Race","CCI"]:
        obj=registry[f"PHYSAPP_HET_{name}_30d"];d=obj["df_fit"]
        profile=profile_builder(d,obj,include_month_year=True)
        raw="physapp_msgs_past_30d"
        settings={"subgroup":name,"focal_column":raw,"focal_log_column":obj["x_phys"],
            "focal_minimum":float(d[raw].min()),"focal_p99":float(d[raw].quantile(.99)),
            "month_year":str(profile["month_year"])}
        export_model(obj,profile,root/name,settings)
    save(root/"COMPLETE.json",{"complete":True,"models":["Sex","Race","CCI"]})
    return root
