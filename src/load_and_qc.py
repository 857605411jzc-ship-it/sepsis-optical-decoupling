from pathlib import Path
from urllib.request import urlopen
import hashlib
import numpy as np
import pandas as pd
import pyreadstat

URL = 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0288249.s001&type=supplementary'
SHA256 = 'ae1f810153b1d75b2d65a4c5f3f411fd0459d377f14f75800409a5a559192cec'
SEED = 2026111234
FIELDS = {'age': 'AGE', 'sex': 'SEX', 'sofa_d1': 'SOFA', 'sofa2': 'SOFA_DAY_2',
          'map_d1': 'MAP_DAY_1', 'map_d2': 'MAP_DAY_2', 'vaso_d1': 'Vasoactive_drugs_use',
          'death': 'Outcome', 'time_to_outcome': 'Time_to_outcome'}
STATES = ['LowB_BelowExpected', 'LowB_AboveExpected', 'HighB_BelowExpected', 'HighB_AboveExpected']


def load(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError('Downloaded source checksum differs')
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Source checksum differs')
    raw, _ = pyreadstat.read_sav(path)
    d = raw[list(FIELDS.values())].rename(columns={v: k for k, v in FIELDS.items()}).copy()
    d['patient_key'] = [f'D{i:03d}' for i in range(1, len(d) + 1)]
    for day in (1, 2):
        for name, source in [('B', 'Basal_PI'), ('crt_sec_d', 'CRT_Seconds'),
                             ('lactate', 'Lactate_level'), ('norepi', 'Noradreline_dose'), ('vp', 'Vasopressin_use')]:
            d[f'{name}{day}'] = raw[f'{source}_day_{day}']
        for t in range(30, 301, 30):
            key = 'IP_POST_TEST_DAY_1_90_SEC_DAY_2' if (day, t) == (2, 90) else f'IP_POST_TEST_DAY_{day}_{t}_SEC'
            d[f'pi_{t}s_d{day}'] = raw[key]
        columns = [f'B{day}', f'crt_sec_d{day}'] + [f'pi_{t}s_d{day}' for t in range(30, 301, 30)]
        # Recover recorded decimal precision before logarithmic transformation.
        d[columns] = d[columns].round(12)
        d['lactate1'] = d.lactate1.round(12)
        seq = d[[f'pi_{t}s_d{day}' for t in range(30, 301, 30)]]
        valid = seq.notna().all(axis=1) & seq.ge(0).all(axis=1) & d[f'B{day}'].gt(0) & seq.max(axis=1).gt(0)
        d[f'B{day}'] = d[f'B{day}'].where(valid)
        d[f'P{day}'] = seq.max(axis=1).where(valid)
        d[f'x{day}'], d[f'y{day}'] = np.log2(d[f'B{day}']), np.log2(d[f'P{day}'])
        d[f'r{day}'] = d[f'y{day}'] - d[f'x{day}']
        d[f'R{day}'] = 2 ** d[f'r{day}']
        d[f'tpeak{day}'] = pd.Series((seq.to_numpy().argmax(axis=1) + 1) / 10).where(valid)
        d[f'shape{day}'] = np.log2(seq.iloc[:, 5:].mean(axis=1) / seq.iloc[:, :5].mean(axis=1)).where(valid)
        d[f'loglac{day}'] = np.log2(d[f'lactate{day}'].where(d[f'lactate{day}'].gt(0)))
    d['lactate_d1'] = d.lactate1
    d['shock'] = np.where(d.vaso_d1.eq(0), 0, np.where(d.lactate1.notna(), d.lactate1.ge(2), np.nan))
    for term in ('x', 'y', 'r'):
        d['d' + term] = d[term + '2'] - d[term + '1']
    d['dCRT'] = d.crt_sec_d2 - d.crt_sec_d1
    d['dloglac'] = d.loglac2 - d.loglac1
    d['dNE_observed'] = d.norepi2 - d.norepi1
    assert len(d) == 226 and d.death.sum() == 90 and d.B1.notna().all()
    assert d.shock.eq(1).sum() == 70 and d.loc[d.shock.eq(1), 'death'].sum() == 41
    assert d.B2.notna().sum() == 218 and d.loc[d.B2.isna(), 'death'].sum() == 8
    return d


def denominators(d):
    paired = d.dx.notna()
    masks = {'Cohort': d.B1.notna(), 'Paired optical': paired,
             'Paired CRT': paired & d.dCRT.notna(), 'Paired lactate': paired & d.dloglac.notna(),
             'After day 2': paired & d.time_to_outcome.gt(2),
             'Next-day CRT and MAP>=65': d.map_d1.ge(65) & d.crt_sec_d2.notna(), 'Shock': d.shock.eq(1)}
    result = pd.DataFrame([{'analysis': k, 'n': int(v.sum()), 'deaths': int(d.loc[v, 'death'].sum())} for k, v in masks.items()])
    assert result.n.tolist() == [226, 218, 217, 175, 208, 211, 70]
    return result
