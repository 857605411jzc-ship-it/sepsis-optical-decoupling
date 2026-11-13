import numpy as np
from io import StringIO
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from .allometric_models import logit
from .load_and_qc import SEED


def validate(d, rng):
    # Use deterministic decimal interchange for validation inputs.
    d = pd.read_csv(StringIO(d.to_csv(index=False)))
    clinical = ['age', 'sofa_d1', 'map_d1', 'crt_sec_d1', 'loglac1', 'vaso_d1']
    models = {'optical_R': ['r1'], 'optical_BE': ['x1', 'E1'],
              'macro_R': ['age', 'map_d1', 'r1'], 'macro_BE': ['age', 'map_d1', 'x1', 'E1'],
              'clinical': clinical, 'clinical_R': clinical+['r1'], 'clinical_B': clinical+['x1'],
              'clinical_BP': clinical+['x1', 'y1'], 'clinical_BE': clinical+['x1', 'E1'],
              'clinical_BR': clinical+['x1', 'r1'], 'clinical_BP_time': clinical+['x1', 'y1', 'tpeak1', 'shape1']}
    rows, differences = [], []
    for task in ['mortality', 'nextday_CRT_MAP65']:
        q = d.copy()
        q['target'] = q.death if task == 'mortality' else q.crt_sec_d2.gt(3).where(q.crt_sec_d2.notna())
        if task != 'mortality':
            q = q[q.map_d1.ge(65)]
        q = q.dropna(subset=['target', 'x1', 'y1']).reset_index(drop=True)
        y, n = q.target.to_numpy(dtype=int), len(q)
        predictions = {k: np.full((n, 20), np.nan) for k in models}
        for repeat in range(20):
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED+repeat)
            for train, test in cv.split(q, y):
                tr, te = q.iloc[train].copy(), q.iloc[test].copy()
                a, b = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr.x1], tr.y1, rcond=None)[0]
                tr['E1'], te['E1'] = tr.y1-a-b*tr.x1, te.y1-a-b*te.x1
                for name, columns in models.items():
                    median = tr[columns].median()
                    xt, xe = tr[columns].fillna(median).to_numpy(), te[columns].fillna(median).to_numpy()
                    f = logit(y[train], xt)
                    predictions[name][test, repeat] = expit(np.c_[np.ones(len(test)), xe] @ f['beta'])
        counts = np.random.default_rng(SEED+991+(task != 'mortality')).multinomial(n, np.full(n, 1/n), size=5000).astype(float)
        positive, negative = y == 1, y == 0
        denominator = counts[:, positive].sum(axis=1)*counts[:, negative].sum(axis=1)
        intervals, losses, aucs = {}, {}, {}
        for name, values in predictions.items():
            assert np.isfinite(values).all()
            loss = ((values-y[:, None])**2).mean(axis=1)
            pair = values[positive, :, None]-values[negative, :].T[None, :, :]
            concordance = ((pair > 0)+.5*(pair == 0)).mean(axis=1)
            bc = counts @ loss/n
            ac = ((counts[:, positive] @ concordance)*counts[:, negative]).sum(axis=1)/denominator
            auc = np.mean([roc_auc_score(y, values[:, i]) for i in range(20)])
            intervals[name], losses[name], aucs[name] = (bc, ac), loss, auc
            rows.append(dict(task=task, model=name, n=n, events=int(y.sum()), auc=auc, brier=loss.mean(),
                logloss=np.mean([log_loss(y, values[:, i]) for i in range(20)]), brier_lower95=np.quantile(bc, .025),
                brier_upper95=np.quantile(bc, .975), auc_lower95=np.quantile(ac, .025), auc_upper95=np.quantile(ac, .975)))
        clinical_intervals = {}
        for base, augmented in [('clinical', 'clinical_BP'), ('clinical_R', 'clinical_BP'),
                                ('clinical_B', 'clinical_BP'), ('clinical_BP', 'clinical_BP_time')]:
            loss = losses[base]-losses[augmented]
            samples = loss[rng.integers(0, n, (5000, n))].mean(axis=1)
            clinical_intervals[(base, augmented)] = np.quantile(samples, [.025, .975])
        for base, augmented in [('optical_R', 'optical_BE'), ('macro_R', 'macro_BE'), ('clinical', 'clinical_BP'),
                                ('clinical_R', 'clinical_BP'), ('clinical_B', 'clinical_BP'), ('clinical_BP', 'clinical_BP_time')]:
            bc, ac = intervals[base][0]-intervals[augmented][0], intervals[augmented][1]-intervals[base][1]
            low, high = clinical_intervals.get((base, augmented), np.quantile(bc, [.025, .975]))
            differences.append(dict(task=task, base=base, augmented=augmented, Brier_improvement=(losses[base]-losses[augmented]).mean(),
                conditional_Brier_lower95=low, conditional_Brier_upper95=high, AUC_improvement=aucs[augmented]-aucs[base],
                conditional_AUC_lower95=np.quantile(ac, .025), conditional_AUC_upper95=np.quantile(ac, .975)))
        print(task, '20 x 5 validation complete', flush=True)
    return {'prediction_metrics': pd.DataFrame(rows), 'prediction_differences': pd.DataFrame(differences)}


def sampling(d):
    sequence = d[[f'pi_{t}s_d1' for t in range(30, 301, 30)]].to_numpy()
    peak = sequence.max(axis=1)
    rows = []
    for step in [1, 2, 3]:
        retained = np.stack([sequence[:, offset::step].max(axis=1) for offset in range(step)], axis=1)
        loss = (100*(1-retained/peak[:, None])).mean(axis=1)
        rows.append(dict(interval_s=30*step, n=len(loss), mean=loss.mean(), median=np.median(loss),
                         q25=np.quantile(loss, .25), q75=np.quantile(loss, .75), p90=np.quantile(loss, .9), p95=np.quantile(loss, .95)))
    return pd.DataFrame(rows)


def coupling(log_b, log_p, beta):
    return log_p-log_b, log_p-beta*log_b
