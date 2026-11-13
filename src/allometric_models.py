import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests
from .load_and_qc import SEED, STATES


def ols(y, X, cluster=None):
    y, X = np.asarray(y, float), np.asarray(X, float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    if cluster is None:
        h = np.sum((X @ bread) * X, axis=1)
        scores = X * (u / np.maximum(1 - h, 1e-8))[:, None]
        covariance = bread @ (scores.T @ scores) @ bread
        df = n - k
    else:
        groups = np.unique(cluster)
        scores = np.array([(X * u[:, None])[cluster == c].sum(axis=0) for c in groups])
        g = len(groups)
        covariance = bread @ (scores.T @ scores) @ bread * g / (g - 1) * (n - 1) / (n - k)
        df = g - 1
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    radius = stats.t.ppf(.975, df) * se
    return dict(beta=b, lo=b-radius, hi=b+radius, p=2*stats.t.sf(abs(b/se), df),
                se=se, df=df, r2=1-u@u/np.sum((y-y.mean())**2), cov=covariance)


def logit(y, X):
    y, X = np.asarray(y, float), np.asarray(X, float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=4000, tol=1e-10).fit((X-mu)/sd, y)
    if model.n_iter_[0] >= 4000:
        raise RuntimeError('Logistic fit did not converge')
    slopes = model.coef_[0] / sd
    b = np.r_[model.intercept_[0] - mu @ slopes, slopes]
    design = np.c_[np.ones(len(y)), X]
    p = expit(design @ b)
    covariance = np.linalg.pinv(design.T @ (design * (p*(1-p))[:, None]))
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    return dict(beta=b, lo=b-1.96*se, hi=b+1.96*se, p=2*stats.norm.sf(abs(b/se)), pred=p)


def regions(d, cut):
    return pd.Series(np.where(d.B1 < cut, 'LowB', 'HighB'), index=d.index) + '_' + np.where(d.E1 < 0, 'BelowExpected', 'AboveExpected')


def profile(d, cut):
    rows = []
    for state, q in d.groupby(regions(d, cut)):
        n, k = len(q), int(q.death.sum())
        z, p = stats.norm.ppf(.975), k/n
        center = (p+z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        row = dict(state=state, n=n, deaths=k, mortality=p, mortality_lower95=center-half, mortality_upper95=center+half)
        for term in ['age', 'sofa_d1', 'map_d1', 'B1', 'P1', 'R1', 'E1', 'crt_sec_d1', 'lactate1', 'norepi1']:
            v = q[term].dropna()
            row.update({term+'_n': len(v), term+'_median': v.median(), term+'_q25': v.quantile(.25), term+'_q75': v.quantile(.75)})
        for term in ['shock', 'vaso_d1']:
            row.update({term+'_yes': int(q[term].eq(1).sum()), term+'_known': int(q[term].notna().sum())})
        rows.append(row)
    return pd.DataFrame(rows).set_index('state').loc[STATES].reset_index()


def analyze(d):
    rng, daily_rng = np.random.default_rng(SEED), np.random.default_rng(SEED+993)
    tables, estimates, deming, intervals = {}, [], [], []
    for day in (1, 2):
        q = d.dropna(subset=[f'x{day}', f'y{day}'])
        x, y = q[f'x{day}'].to_numpy(), q[f'y{day}'].to_numpy()
        f = ols(y, np.c_[np.ones(len(q)), x])
        curved = ols(y, np.c_[np.ones(len(q)), x, (x-x.mean())**2])
        estimates.append(dict(day=day, n=len(q), alpha=f['beta'][0], alpha_lo=f['lo'][0], alpha_hi=f['hi'][0],
                              beta=f['beta'][1], lower95=f['lo'][1], upper95=f['hi'][1], r2=f['r2'],
                              p_beta_equals_one=2*stats.t.sf(abs((f['beta'][1]-1)/f['se'][1]), f['df']),
                              quadratic=curved['beta'][2], quadratic_p=curved['p'][2]))
        ix = daily_rng.integers(0, len(q), (5000, len(q)))
        xx, yy = x[ix], y[ix]
        xc, yc = xx-xx.mean(axis=1)[:, None], yy-yy.mean(axis=1)[:, None]
        slopes = (xc*yc).sum(axis=1)/(xc*xc).sum(axis=1)
        intercepts = yy.mean(axis=1)-slopes*xx.mean(axis=1)
        intervals.append(dict(day=day, bootstrap_lower95=np.quantile(slopes, .025), bootstrap_upper95=np.quantile(slopes, .975),
                              alpha_lower95=np.quantile(intercepts, .025), alpha_upper95=np.quantile(intercepts, .975)))
        sxx, syy, sxy = np.var(x, ddof=1), np.var(y, ddof=1), np.cov(x, y)[0, 1]
        for ratio in (.25, 1, 4):
            slope = (syy-ratio*sxx+np.sqrt((syy-ratio*sxx)**2+4*ratio*sxy*sxy))/(2*sxy)
            deming.append(dict(day=day, error_ratio=ratio, beta_deming=slope))
    a, b = estimates[0]['alpha'], estimates[0]['beta']
    for day in (1, 2):
        d[f'E{day}'] = d[f'y{day}']-a-b*d[f'x{day}']
        d[f'A{day}'] = 2**d[f'E{day}']
        d[f'state{day}'] = np.where(d[f'B{day}'] < d.B1.median(), 'LowB', 'HighB') + np.where(d[f'E{day}'] < 0, '_BelowExpected', '_AboveExpected')
        d.loc[d[f'B{day}'].isna(), f'state{day}'] = np.nan
    d['dE'] = d.E2-d.E1
    q = d.dropna(subset=['x1', 'x2', 'y1', 'y2'])
    delta = []
    for _ in range(5000):
        s = q.iloc[rng.integers(0, len(q), len(q))]
        delta.append(np.polyfit(s.x2, s.y2, 1)[0]-np.polyfit(s.x1, s.y1, 1)[0])
    stacked = pd.concat([pd.DataFrame({'id': d.patient_key, 'x': d[f'x{day}'], 'y': d[f'y{day}'], 'day2': day-1}) for day in (1, 2)]).dropna()
    f = ols(stacked.y, np.c_[np.ones(len(stacked)), stacked.x, stacked.day2, stacked.x*stacked.day2], stacked.id.to_numpy())
    paired_beta = [np.polyfit(q[f'x{day}'], q[f'y{day}'], 1)[0] for day in (1, 2)]
    tables['day_interaction'] = pd.DataFrame([dict(paired_n=len(q), paired_day1=paired_beta[0], paired_day2=paired_beta[1],
        difference=paired_beta[1]-paired_beta[0], bootstrap_lo=np.quantile(delta, .025), bootstrap_hi=np.quantile(delta, .975),
        stacked_n=len(stacked), cluster_n=stacked.id.nunique(), interaction=f['beta'][3], lo=f['lo'][3], hi=f['hi'][3], p=f['p'][3])])
    models = []
    def fit(name, target, features, linear=False, frame=d):
        q = frame.dropna(subset=[target]+features)
        f = ols(q[target], np.c_[np.ones(len(q)), q[features]]) if linear else logit(q[target], q[features])
        transform = (lambda x: x) if linear else np.exp
        for j, term in enumerate(['intercept']+features):
            models.append(dict(model=name, term=term, n=len(q), events=np.nan if linear else int(q[target].sum()),
                coefficient=f['beta'][j], coef_lo=f['lo'][j], coef_hi=f['hi'][j], effect=transform(f['beta'][j]),
                lower95=transform(f['lo'][j]), upper95=transform(f['hi'][j]), p=f['p'][j]))
    for columns in [['r1'], ['x1', 'r1']]:
        fit('mortality_unadjusted_'+'_'.join(columns), 'death', columns)
    base = ['age', 'sofa_d1']
    fit('mortality_minimal_BE', 'death', base+['x1', 'E1'])
    fit('mortality_extended_BE', 'death', base+['map_d1', 'loglac1', 'vaso_d1', 'x1', 'E1'])
    fit('mortality_shock_1', 'death', base+['x1', 'E1'], frame=d[d.shock.eq(1)])
    for label, target in [('CRT', 'crt_sec_d1'), ('lactate', 'loglac1')]:
        fit('anchor_'+label, target, base+['map_d1', 'x1', 'E1'], True)
    for label, target, initial in [('CRT', 'crt_sec_d2', 'crt_sec_d1'), ('lactate', 'loglac2', 'loglac1')]:
        fit('day2_'+label+'_trajectory', target, base+[initial, 'x1', 'E1', 'dx', 'dE'], True)
    features = base+['x1', 'E1', 'dx', 'dE']
    fit('mortality_trajectory_landmark_day2', 'death', features, frame=d[d.time_to_outcome.gt(2)])
    tables['associations'] = pd.DataFrame(models)
    family = tables['associations'].query('term in ["dx", "dE"]').copy()
    family['q'] = multipletests(family.p, method='fdr_bh')[1]
    tables['trajectory_family'] = family
    for label, columns in [('mortality_BE', base+['x1', 'E1']), ('mortality_trajectory', features)]:
        draws = []
        for _ in range(1500):
            s = d.iloc[rng.integers(0, len(d), len(d))].copy()
            aa, bb = np.linalg.lstsq(np.c_[np.ones(len(s)), s.x1], s.y1, rcond=None)[0]
            s['E1'], s['E2'] = s.y1-aa-bb*s.x1, s.y2-aa-bb*s.x2
            s['dE'] = s.E2-s.E1
            if label == 'mortality_trajectory':
                s = s[s.time_to_outcome.gt(2)].dropna(subset=columns)
            draws.append(logit(s.death, s[columns])['beta'][-2:])
        limits = np.exp(np.quantile(draws, [.025, .975], axis=0))
        tables[label+'_refit_bootstrap'] = pd.DataFrame({'term': columns[-2:], 'OR_lower95': limits[0], 'OR_upper95': limits[1], 'replicates': 1500})
    q = d[d.time_to_outcome.gt(2)].dropna(subset=features)
    loo = [np.exp(logit(q.drop(i).death, q.drop(i)[features])['beta'][-1]) for i in q.index]
    tables['trajectory_leave_one_out'] = pd.DataFrame({'removed': q.patient_key, 'OR_dE': loo})
    discordant = d[d.dx.gt(0) & d.dy.gt(0) & d.dr.lt(0)]
    tables['discordant'] = pd.DataFrame([dict(measure=c, n=len(v := discordant[c].dropna()), decreased=int((v < -1e-10).sum()),
        unchanged=int((abs(v) <= 1e-10).sum()), increased=int((v > 1e-10).sum())) for c in ['dx', 'dy', 'dr', 'dE', 'dCRT', 'dloglac', 'dNE_observed']])
    for i, pair in enumerate([['B1','B2'], ['P1','P2'], ['R1','R2'], ['E1','E2'], ['crt_sec_d1','crt_sec_d2'], ['lactate1','lactate2'], ['norepi1','norepi2']]):
        values = discordant[pair].dropna().quantile([.25, .5, .75]).to_numpy().T.ravel()
        tables['discordant'].loc[i, ['day1_q25','day1_median','day1_q75','day2_q25','day2_median','day2_q75']] = values
    for name, rows in [('allometry', estimates), ('allometry_bootstrap', intervals), ('deming_sensitivity', deming)]:
        tables[name] = pd.DataFrame(rows)
    tables['day1_states'], tables['cutoff_1_4'] = profile(d, d.B1.median()), profile(d, 1.4)
    return d, tables, rng
