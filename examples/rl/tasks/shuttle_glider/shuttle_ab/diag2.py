import pandas as pd, numpy as np, sys
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'play_records/probe/trajectories.csv')
KF5, WMAX5 = 8.54858e-06, 1000.0
FMAX = KF5 * WMAX5**2
for ctrl, g in d.groupby('controller'):
    g = g[g.episode_step.between(200, 990)]
    w5 = g['actual_w_4'].to_numpy(); F5 = KF5 * w5**2
    print('=' * 62); print(ctrl)
    print('  w5  p05=%.0f p25=%.0f p50=%.0f p75=%.0f p95=%.0f   (cap %.0f)'
          % (*np.percentile(w5, [5, 25, 50, 75, 95]), WMAX5))
    print('  tempo com w5 < 50 rad/s : %5.1f %%' % (100 * (w5 < 50).mean()))
    print('  tempo com w5 > 950      : %5.1f %%' % (100 * (w5 > 950).mean()))
    print('  thrust5 media %.2f N  sd %.2f  max %.2f   (disponivel %.2f N)'
          % (F5.mean(), F5.std(), F5.max(), FMAX))
    print('  -> usa %.0f %% da autoridade; inclinacao equivalente %.1f graus'
          % (100 * F5.mean() / FMAX, np.degrees(np.arctan(F5.mean() / 48.8))))
    vref = np.linalg.norm(g[['ref_vx', 'ref_vy']].to_numpy(), axis=1)
    pairs = [('|v_ref_xy|', vref)] + [(c, g[c].to_numpy()) for c in
             ('ty_body', 'tx_body', 'pos_error', 'speed', 'att_err_pd') if c in g]
    for name, v in pairs:
        if np.std(v) > 1e-9 and F5.std() > 1e-9:
            print('  corr(thrust5, %-12s) = %+.3f' % (name, np.corrcoef(F5, v)[0, 1]))