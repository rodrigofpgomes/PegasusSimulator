import pandas as pd, numpy as np, sys
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'play_records/probe/trajectories.csv')

wcol = [c for c in d.columns if c.startswith('actual_w')] or \
       [c for c in d.columns if c.startswith('cmd_w')]
wcol = sorted(wcol, key=lambda c: int(''.join(ch for ch in c if ch.isdigit())))
KF = np.array([1.709716e-05]*4 + [8.54858e-06])[:len(wcol)]

print('rotores:', wcol)
print('ref_vx |max| = %.3f   ref_vy |max| = %.3f' % (d.ref_vx.abs().max(), d.ref_vy.abs().max()))
print('speed max = %.3f' % d.speed.max())

for ctrl, g in d.groupby('controller'):
    print('=' * 62); print(ctrl)
    prof = g.groupby('episode_step').pos_error.median()
    for s in [0, 25, 50, 100, 200, 400, 700, 999]:
        if s in prof.index:
            print('   t=%5.2f s   pos_error mediano = %.3f m' % (s * 0.01, prof.loc[s]))
    gs = g[g.episode_step >= 200]          # descarta os primeiros 2 s
    print('   RMSE_pos episodio inteiro = %.3f m' % np.sqrt((g.pos_error**2).mean()))
    print('   RMSE_pos apos 2 s         = %.3f m' % np.sqrt((gs.pos_error**2).mean()))
    print('   pos_error p95 apos 2 s    = %.3f m' % gs.pos_error.quantile(0.95))
    if len(wcol) == 5:
        F = KF[4] * gs[wcol[4]].to_numpy()**2
        Ftot = (KF * gs[wcol].to_numpy()**2).sum(axis=1)
        vref = np.linalg.norm(gs[['ref_vx', 'ref_vy']].to_numpy(), axis=1)
        print('   thrust5 medio = %.2f N  (%.1f %% do total)' % (F.mean(), 100*F.mean()/Ftot.mean()))
        print('   w5 medio = %.0f rad/s   max = %.0f  (cap 3500)' % (gs[wcol[4]].mean(), gs[wcol[4]].max()))
        if F.std() > 1e-9 and vref.std() > 1e-9:
            print('   corr(thrust5, |v_ref_xy|) = %+.3f' % np.corrcoef(F, vref)[0, 1])