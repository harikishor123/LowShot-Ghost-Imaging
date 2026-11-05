"""Print the audit report's summary tables directly from the frozen result files (read-only)."""
import json, pandas as pd
E='outputs/extension_medmnist_microscopy/results/'; P='outputs/extension_medmnist_microscopy_poisson/results/'
O1='outputs/phase_final_low_shot_to_10k_reference/'; O2='outputs/phase_final_original_target/results/'
def n(x): return 'none ≤ 5,000' if x is None else f'{x:,}'
def J(p): return json.load(open(p))
print('| Experiment | Method / comparator | All (val → test) | Blood (val → test) | Tissue (val → test) |'); print('|---|---|---|---|---|')
print(f"| Onion Phase 1 | AI vs independent 10k | {n(J(O1+'validation/evaluation_report.json')['minimum_successful_shot_count_AI'])} → {n(J(O1+'test/evaluation_report.json')['minimum_successful_shot_count_AI'])} | — | — |")
print(f"| Onion Phase 2 | AI-original vs DGI-10000 | {n(J(O2+'val/evaluation_report.json')['minimum_high_fidelity_shot_count'])} → {n(J(O2+'test/evaluation_report.json')['minimum_high_fidelity_shot_count'])} | — | — |")
spec=[('Extension, clean target',E+'target_clean','minimum_passing_shot_count','AI-original','AI-original vs DGI-10000'),
      ('Extension, 10k-DGI target',E+'target_dgi10k','minimum_passing_shot_count','AI','AI vs independent 10k'),
      ('Extension, 10k-DGI target',E+'target_dgi10k','minimum_passing_shot_count','smoothed DGI','smoothed DGI vs independent 10k'),
      ('Extension, 10k-DGI target',E+'target_dgi10k','minimum_passing_shot_count','DGI','conventional DGI vs independent 10k')]
for l,nm in (('p05','5%'),('p10','10%'),('p30','30%')):
    spec+=[(f'Poisson {nm}',P+f'level_{l}','minimum_passing_shot_count','AI-original','AI-original vs noisy DGI-10000'),
           (f'Poisson {nm}',P+f'level_{l}','minimum_passing_shot_count','smoothed DGI','smoothed DGI vs noisy DGI-10000'),
           (f'Poisson {nm}',P+f'level_{l}','minimum_passing_shot_count_vs_noise_free_DGI10000','AI-original','AI-original vs noise-free DGI-10000')]
for e,b,k,m,lab in spec:
    v,t=J(b+'/val/evaluation_report.json')[k],J(b+'/test/evaluation_report.json')[k]
    cells=[]
    for g in ('all','blood','path'):
        a,c=v[g][m],t[g][m]; cells.append(f"{n(a)} → {n(c)}" + (' **(differs)**' if a!=c else ''))
    print(f'| {e} | {lab} | '+' | '.join(cells)+' |')
