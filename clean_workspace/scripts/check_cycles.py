import json
from core.pipeline.pipeline import run_pipeline_clean

if __name__ == "__main__":
    with open("imu-gait-diagnostics/tools/kwargs.json", "r") as f:
        kw = json.load(f)
    out = run_pipeline_clean(**kw)
    ev = out['events']
    print({k: len(v) for k, v in ev.items()})
    L = out['cycles']['L']; R = out['cycles']['R']
    print({"L_used": L['count_used'], "L_total": L['count_total'], "R_used": R['count_used'], "R_total": R['count_total']})
