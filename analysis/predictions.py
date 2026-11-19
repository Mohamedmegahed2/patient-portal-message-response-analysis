"""Conditional prediction contrasts using full patient-clustered covariance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from .common import latest_run,new_run,finish,save,sha
Z=norm.ppf(.975)
GROUPS={
    "Sex":("Gender",["Male","Female"]),
    "Race":("Race",["Asian","Black/African American","Other/Multiple/Unknown","White"]),
    "CCI":("Charlson_group",["5+ (Severe)","3-4 (Moderate)","1-2 (Mild)","0 (None)"])}
def design_vector(terms, profile, overrides):
    """Reconstruct saved named columns, including categorical and interaction terms."""
    values = {**profile, **overrides}
    def factor(term):
        if term == "Intercept":
            return 1.0
        if "[T." in term:
            prefix, level = term.split("[T.", 1)
            assert level.endswith("]")
            variable = prefix[2:].split(",", 1)[0].rstrip(")") if prefix.startswith("C(") else prefix
            assert variable in ["Gender", "Race", "Ethnicity", "Charlson_group", "month_year"]
            return float(str(values[variable]) == level[:-1])
        if term not in values:
            raise ValueError(f"Missing saved profile factor: {term}")
        return float(values[term])
    return np.array([np.prod([factor(part) for part in term.split(":")]) for term in terms], dtype=float)

class AggregateModel:
    def __init__(self,path):
        self.path=Path(path)
        for name,h in json.loads((self.path/"manifest.json").read_text(encoding="utf-8")).items():
            assert Path(name).name==name and sha(self.path/name)==h
        c=pd.read_csv(self.path/"coefficients.csv")
        self.terms=c.term.tolist();self.beta=c.beta.to_numpy()
        cov=pd.read_csv(self.path/"patient_clustered_covariance.csv",index_col=0)
        self.cov=cov.loc[self.terms,self.terms].to_numpy()
        self.profile=json.loads((self.path/"profile.json").read_text(encoding="utf-8"))
        self.settings=json.loads((self.path/"settings.json").read_text(encoding="utf-8"))
        assert self.settings["converged"] and np.isfinite(self.beta).all() and np.isfinite(self.cov).all()
        assert np.allclose(self.cov,self.cov.T,atol=1e-12)
        assert np.linalg.eigvalsh(self.cov).min()>-1e-10
        assert np.allclose(np.diag(self.cov)**.5,c.cluster_robust_se)
    def predict(self,overrides):
        x=design_vector(self.terms,self.profile,overrides)
        eta=float(x@self.beta);p=float(expit(eta))
        se=float(np.sqrt(max(0.,x@self.cov@x)))
        return {"probability":p,"lower":float(expit(eta-Z*se)),
                "upper":float(expit(eta+Z*se)),"gradient":p*(1-p)*x}
    def difference(self,positive,negative):
        diff=positive["probability"]-negative["probability"]
        gradient=positive["gradient"]-negative["gradient"]
        se=float(np.sqrt(max(0.,gradient@self.cov@gradient)))
        return {"difference_percentage_points":100*diff,
                "ci95_lower":100*(diff-Z*se),"ci95_upper":100*(diff+Z*se),
                "gradient":gradient}

def probability_row(p):
    return {"predicted_probability_pct":100*p["probability"],
            "ci95_lower_pct":100*p["lower"],"ci95_upper_pct":100*p["upper"]}

def compute(primary,subgroups,run):
    curves=[];settings={};profiles={}
    for letter,m in primary.items():
        st=m.settings;settings[letter]=st;profiles[letter]=m.profile
        assert st["focal_minimum"]==(0 if letter=="C" else 1)
        for percentile in [50,90,99]:
            moderator=st["moderator_percentiles"][str(percentile)]
            for count in np.linspace(st["focal_minimum"],st["focal_p99"],250):
                p=m.predict({st["focal_log_column"]:np.log1p(count),
                             st["moderator_log_column"]:np.log1p(moderator)})
                curves.append({"panel":letter,"message_count_30d":count,
                    "moderator_percentile":percentile,"moderator_count":moderator,**probability_row(p)})
    f1=pd.DataFrame(curves);f1.to_csv(run/"figure1_curves_with_pointwise_ci.csv",index=False)
    save(run/"figure1_model_settings.json",settings);save(run/"figure1_prediction_profiles.json",profiles)
    probabilities=[];differences=[]
    specs=[("M1_HCW","A","provider_msgs_past_30d",(2,8),"system_msgs_past_30d",0),
           ("M2_PhysAPP","B","physapp_msgs_past_30d",(1,5),"other_human_msgs_past_30d",1),
           ("M3_OtherHCW","C","other_human_msgs_past_30d",(2,5),"physapp_msgs_past_30d",0),
           ("M3_PhysAPP","C","physapp_msgs_past_30d",(0,3),"other_human_msgs_past_30d",2)]
    for name,letter,column,levels,other,held in specs:
        m=primary[letter]
        pair=[m.predict({"log_"+column:np.log1p(c),"log_"+other:np.log1p(held)}) for c in levels]
        for c,p in zip(levels,pair):
            probabilities.append({"contrast":name,"message_count":c,"held_exposure":other,"held_count":held,**probability_row(p)})
        diff=m.difference(pair[1],pair[0]);diff.pop("gradient")
        differences.append({"contrast":name,"lower_count":levels[0],"higher_count":levels[1],**diff})
    pd.DataFrame(probabilities).to_csv(run/"primary_model_probabilities.csv",index=False)
    pd.DataFrame(differences).to_csv(run/"primary_model_probability_differences.csv",index=False)
    curves=[];probabilities=[];differences=[];changes=[];profiles={}
    for panel,(col,groups) in GROUPS.items():
        m=subgroups[panel];st=m.settings;profiles[panel]=m.profile
        assert st["focal_minimum"]==1 and st["focal_p99"]==13
        for group in groups:
            for count in np.linspace(1,13,241):
                p=m.predict({st["focal_log_column"]:np.log1p(count),col:group})
                curves.append({"panel":panel,"group":group,"message_count_30d":count,**probability_row(p)})
        # Only reported race and moderate-comorbidity pairwise comparisons.
        if panel=="Sex":continue
        positive,reference=("Black/African American","White") if panel=="Race" else ("3-4 (Moderate)","0 (None)")
        ds=[]
        for count in [1,5]:
            pair=[m.predict({st["focal_log_column"]:np.log1p(count),col:g}) for g in [positive,reference]]
            for group,p in zip([positive,reference],pair):
                probabilities.append({"panel":panel,"group":group,"message_count":count,**probability_row(p)})
            diff=m.difference(pair[0],pair[1]);ds.append(diff)
            differences.append({"panel":panel,"comparison_group":positive,"reference_group":reference,
                "message_count":count,**{k:v for k,v in diff.items() if k!="gradient"}})
        delta=(ds[1]["difference_percentage_points"]-ds[0]["difference_percentage_points"])/100
        gradient=ds[1]["gradient"]-ds[0]["gradient"]
        se=float(np.sqrt(max(0.,gradient@m.cov@gradient)))
        changes.append({"panel":panel,"change_in_difference_percentage_points":100*delta,
            "ci95_lower":100*(delta-Z*se),"ci95_upper":100*(delta+Z*se)})
    f2=pd.DataFrame(curves);f2.to_csv(run/"figure2_curves_with_pointwise_ci.csv",index=False)
    pd.DataFrame(probabilities).to_csv(run/"subgroup_probabilities_at_1_and_5.csv",index=False)
    pd.DataFrame(differences).to_csv(run/"subgroup_probability_differences.csv",index=False)
    pd.DataFrame(changes).to_csv(run/"subgroup_change_in_differences.csv",index=False)
    save(run/"figure2_prediction_profiles.json",profiles)
    from .plotting import figure1,figure2
    figure1(f1,settings,run);figure2(f2,run)
    return run

def run_analysis():
    primary=latest_run("analysis_models_",["aggregate_models/COMPLETE.json"])/"aggregate_models"
    subgroup=latest_run("physapp_demographic_cci_interactions_",["aggregate_models/COMPLETE.json"])/"aggregate_models"
    run=new_run("prediction_contrasts")
    save(run/"input_models.json",{"primary":str(primary),"subgroups":str(subgroup)})
    compute({x:AggregateModel(primary/x) for x in "ABC"},
            {x:AggregateModel(subgroup/x) for x in GROUPS},run)
    return finish(run)
