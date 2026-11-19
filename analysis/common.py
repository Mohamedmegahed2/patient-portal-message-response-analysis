from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, uuid, platform
import importlib.metadata
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("JMIR_REPRO_OUTPUT_DIR", ROOT / "outputs")).expanduser().resolve()
WORKING = Path(os.environ.get("JMIR_WORKING_DATA_DIR", OUTPUT_ROOT / "working_data")).expanduser().resolve()
PRIMARY = WORKING / "analysis_df_all_exposures_with_encounter_covariates.parquet"
MESSAGES = WORKING / "messages_with_threads.parquet"
ENCOUNTERS = WORKING / "encounters_unified_clean.parquet"
def require(*paths):
    missing=[str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Run the preceding preparation notebooks with authorized local data. Missing: " + "; ".join(missing))
def latest_run(prefix, required=()):
    candidates=sorted((OUTPUT_ROOT / "runs").glob(prefix+"*"),reverse=True)
    for p in candidates:
        if p.is_dir() and all((p/n).is_file() for n in required):return p
    raise FileNotFoundError(f"No completed {prefix} run under {OUTPUT_ROOT / 'runs'}; run the preceding notebook first.")
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def save(path,value):
    def default(v):
        if hasattr(v,"item"):return v.item()
        return str(v)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=default)+"\n",encoding="utf-8")
def new_run(prefix, inputs=()):
    require(*inputs)
    p=OUTPUT_ROOT/"runs"/(prefix+"_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+"_"+uuid.uuid4().hex[:6])
    p.mkdir(parents=True,exist_ok=False)
    versions={}
    for name in ["numpy","pandas","scipy","statsmodels","patsy","numba","matplotlib","pyarrow"]:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    save(p/"run_manifest.json",{"created_utc":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),"packages":versions,
        "input_sha256":{Path(x).name:sha(x) for x in inputs}})
    return p
def finish(run):
    save(run/"output_manifest.json",{p.name:sha(p) for p in run.iterdir()
        if p.is_file() and p.name not in ["output_manifest.json","COMPLETE.json"]})
    save(run/"COMPLETE.json",{"complete":True,"created_utc":datetime.now(timezone.utc).isoformat()})
    return run
