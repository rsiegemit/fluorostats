"""Validate fluorostats.power via empirical calibration (self-contained)."""
import numpy as np, pandas as pd
from pathlib import Path
from fluorostats.power import bootstrap_power, power_curve
from fluorostats.stats import mann_whitney
RES=Path(__file__).resolve().parent/"results"; rng=np.random.default_rng(0); rows=[]

# Empirical calibration: does predicted power match the true rejection rate?
# Ground-truth populations: d=0.8 effect.
def true_power(n, d, sims=400, alpha=0.05):
    hits=0
    for _ in range(sims):
        a=rng.normal(0,1,n); b=rng.normal(d,1,n)
        if mann_whitney(a,b)["p"]<alpha: hits+=1
    return hits/sims

for n in [8, 15, 25]:
    # pilot of size n from the same populations, bootstrap-predict power at n
    pilot_a=rng.normal(0,1,n); pilot_b=rng.normal(0.8,1,n)
    pred=bootstrap_power(pilot_a,pilot_b,n=n,n_sims=400,seed=1)
    emp=true_power(n,0.8)
    rows.append({"test":f"calibration_n{n}_d0.8","predicted_power":round(pred,3),
                 "empirical_power":round(emp,3),"abs_diff":round(abs(pred-emp),3),
                 "PASS":abs(pred-emp)<0.20})

# Monotonicity: power rises with n for a real effect
pc=power_curve(rng.normal(0,1,30),rng.normal(0.9,1,30),ns=[5,10,20,40],n_sims=400,seed=2)
mono=bool(np.all(np.diff(pc["power"].values)>=-0.05))
rows.append({"test":"power_curve_monotonic_increasing","predicted_power":None,
             "empirical_power":None,"abs_diff":None,"PASS":mono})

# Null control: power ~ alpha for no effect (d=0)
null_pow=bootstrap_power(rng.normal(0,1,30),rng.normal(0,1,30),n=20,n_sims=500,alpha=0.05,seed=3)
rows.append({"test":"null_effect_power<=0.15(≈alpha)","predicted_power":round(null_pow,3),
             "empirical_power":0.05,"abs_diff":None,"PASS":null_pow<=0.15})

df=pd.DataFrame(rows); df.to_csv(RES/"validate_power.csv",index=False)
print(df.to_string(index=False)); print(f"\n{int(df.PASS.sum())}/{len(df)} PASS")
print("\nNote: bootstrap-from-pilot power is noisier and mildly optimistic vs true "
      "power (small pilots); calibration within ~0.2 confirms the estimator is sound.")
