import re, pandas as pd
T=open('thesis/MTP_Thesis_Final.md').read().replace('−','-')
P1='outputs/phase_final_low_shot_to_10k_reference/test/'; P2='outputs/phase_final_original_target/results/test/'
s=pd.read_csv(P1+'summary.csv'); c=pd.read_csv(P1+'comparability.csv')
ag=pd.read_csv(P2+'aggregate_metrics.csv'); av=pd.read_csv(P2+'ai_vs_dgi_comparison.csv')
ALL='all eval seeds (2026-2029)'
def S(m,n,met,st='mean',sg=ALL): return float(s[(s.seed_group==sg)&(s.method==m)&(s.shot_count==n)][f'{met}_{st}'].iloc[0])
def Cc(m,n,col): return c[(c.method==m)&(c.shot_count==n)][col].iloc[0]
def f(x,d=3): return f'{x:.{d}f}'
def sg(x,d=3): return f'{x:+.{d}f}'
def rowcheck(name, label_regex, expected_tokens):
    m=re.search(r'^\|\s*'+label_regex+r'.*$', T, re.M)
    if not m: print('ROW NOT FOUND', name); return 1
    nums=re.findall(r'[+-]?\d[\d,]*\.\d+|[+-]?\d+%', m.group(0))
    exp=[t for t in expected_tokens]
    got=[x for x in nums]
    bad=[(e,g) for e,g in zip(exp,got) if e.lstrip('+')!=g.lstrip('+')]
    if len(exp)!=len(got) or bad: print('MISMATCH', name, '| thesis:', got, '| csv:', exp); return 1
    return 0
bad=0; n=0
# Table 4.1
for N,lab in [(500,'AI-500'),(1000,'AI-1000'),(2000,'AI-2000'),(5000,'AI-5000')]:
    n+=1; bad+=rowcheck('T4.1 '+lab, re.escape(lab)+r'\s*\|\s*[\d,]+', [f(S('AI',N,'SSIM')),f(S('AI',N,'SSIM','std')),f(S('AI',N,'correlation')),f(S('AI',N,'correlation','std')),f(S('AI',N,'PSNR'),2),f(S('AI',N,'PSNR','std'),2)])
n+=1; bad+=rowcheck('T4.1 retest','Independent 10,000-shot acquisition',[f(S('independent 10k (retest)',10000,x,st),d) for x,st,d in [('SSIM','mean',3),('SSIM','std',3),('correlation','mean',3),('correlation','std',3),('PSNR','mean',2),('PSNR','std',2)]])
# Table 4.2
for N in [500,1000,2000,5000]:
    n+=1; bad+=rowcheck(f'T4.2 AI-{N}', rf'AI-{N}\s*\|\s*[+-]', [sg(Cc('AI',N,'SSIM_diff_vs_retest_mean')),sg(Cc('AI',N,'SSIM_ci_low')),sg(Cc('AI',N,'SSIM_ci_high')),sg(Cc('AI',N,'correlation_diff_vs_retest_mean')),sg(Cc('AI',N,'correlation_ci_low')),sg(Cc('AI',N,'correlation_ci_high'))])
# Table 4.3
for N in [500,1000,2000,5000]:
    for m,lab in [('conventional DGI','Conventional DGI'),('smoothed DGI','Gaussian-smoothed DGI')]:
        n+=1; bad+=rowcheck(f'T4.3 {lab} {N}', rf'{N:,}\s*\|\s*{lab}', [f(S(m,N,'SSIM')),f(S(m,N,'correlation')),f(S(m,N,'PSNR'),2)])
# Table 4.4
for N in [500,1000,2000,5000]:
    dp=S('AI',N,'PSNR','mean','existing reference (2026)')-S('AI',N,'PSNR','mean','unseen sequences (2027-2029)')
    ds=S('AI',N,'SSIM','mean','existing reference (2026)')-S('AI',N,'SSIM','mean','unseen sequences (2027-2029)')
    n+=1; bad+=rowcheck(f'T4.4 AI-{N}', rf'AI-{N}\s*\|\s*-?\d+\.\d+ dB', [f(dp),f(ds)])
# Table 4.5
n+=1; bad+=rowcheck('T4.5 retest','Independent 10k acquisition',[f(S('independent 10k (retest)',10000,'edge_r'))])
for N in [500,1000,2000,5000]:
    n+=1; bad+=rowcheck(f'T4.5 AI-{N}', rf'AI-{N}\s*\|\s*0\.\d+\s*\|', [f(S('AI',N,'edge_r')),sg(Cc('AI',N,'edge_r_diff_vs_retest_mean')),sg(Cc('AI',N,'edge_r_ci_low')),sg(Cc('AI',N,'edge_r_ci_high')),f"{Cc('AI',N,'edge_r_pct_pairs_ge_retest'):.0f}%"])
# Table 4.7
def G(lab,met,st='mean'): return float(ag[(ag.seed_group=='all seeds (2026-2029)')&(ag.label==lab)][f'{met}_{st}'].iloc[0])
for lab,rx in [(f'{m}-{N}', (r'\*\*AI-original-%d\*\*'%N) if m=='AI-original' else (r'Conventional DGI-%d'%N)) for N in [500,1000,2000,5000] for m in ['DGI','AI-original']]+[('DGI-10000',r'Conventional 10k DGI baseline \(DGI-10000\)')]:
    n+=1; bad+=rowcheck('T4.7 '+lab, rx+r'\s*\|\s*[\d,]+', [f(G(lab,'SSIM')),f(G(lab,'SSIM','std')),f(G(lab,'correlation')),f(G(lab,'correlation','std')),f(G(lab,'PSNR'),2),f(G(lab,'PSNR','std'),2),f(G(lab,'MSE'),4),f(G(lab,'MSE','std'),4),f(G(lab,'edge_r')),f(G(lab,'edge_r','std'))])
# Table 4.8
for N in [500,1000,2000,5000]:
    r=av[(av.comparison=='AI-original-N minus DGI-10000')&(av.shot_count==N)].iloc[0]
    tok=[]
    for met,d in [('SSIM',3),('correlation',3),('PSNR',2),('MSE',4),('edge_r',3)]:
        tok+= [sg(r[f'{met}_mean'],d),sg(r[f'{met}_ci_low'],d),sg(r[f'{met}_ci_high'],d),f"{r[f'{met}_pct_A_better']:.0f}%"]
    n+=1; bad+=rowcheck(f'T4.8 {N}', rf'{N:,}\s*\|\s*[+-]0', tok)
print(f'onion thesis rows checked: {n}, mismatching rows: {bad}')
# inline/abstract values
for txt,val in [('SSIM 0.469, r 0.710', (f(S('independent 10k (retest)',10000,'SSIM')),f(S('independent 10k (retest)',10000,'correlation')))),
                ('AI-1000 (SSIM 0.494, r 0.759, PSNR 21.44 dB)',(f(S('AI',1000,'SSIM')),f(S('AI',1000,'correlation')),f(S('AI',1000,'PSNR'),2))),
                ('SSIM 0.611, r 0.842, edge correlation 0.713',(f(G('DGI-10000','SSIM')),f(G('DGI-10000','correlation')),f(G('DGI-10000','edge_r')))),
                ('(SSIM 0.640, r 0.867)',(f(G('AI-original-1000','SSIM')),f(G('AI-original-1000','correlation')))),
                ('SSIM 0.817, r 0.941, PSNR 28.22 dB',(f(G('AI-original-5000','SSIM')),f(G('AI-original-5000','correlation')),f(G('AI-original-5000','PSNR'),2)))]:
    print('abstract:', txt, '-> csv', val, 'present in thesis:', txt in T or txt.replace(' r ',' *r* ') in T)
# smoothed DGI 1000 SSIM lower bound quoted in 4.5
print('smoothed DGI-1000 SSIM ci_low', f(Cc('smoothed DGI',1000,'SSIM_ci_low')), '(thesis 4.5 says -0.031)')
