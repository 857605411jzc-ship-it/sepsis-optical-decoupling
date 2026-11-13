# Sepsis optical decoupling

Reproduction code for *Decoupling basal state from reactivity in optical sepsis monitoring*.
The fixed day-1 reference is E = log2(P) - alpha - beta log2(B), with A = 2**E and R = P/B.
One E unit doubles A; one delta E unit doubles A2/A1.

## Reproduce

Use Python 3.12. Pinned dependencies include `pyreadstat` for the original SPSS file.

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The entry point downloads and checks the official source. For an existing file:

```bash
python run_all.py --data /path/to/pone.0288249.s001.sav --out /path/to/results
```

Run from the repository root; `--no-figures` skips rendering. No GPU is required.

## Source

de Miranda AC et al. *PLoS One*. 2023;18:e0288249.
https://doi.org/10.1371/journal.pone.0288249
See `data/README.md` for the official download and SHA256. Participant records are excluded.

## Methods and outputs

- **Allometric scaling & linear models:** Ordinary least squares with HC3 heteroscedasticity-consistent standard errors; repeated-day interaction models use patient-clustered covariance.
- **Mortality models:** Unpenalized maximum-likelihood logistic regression complemented by 1,500 patient-level bootstrap resamples refitting both the allometric reference and outcome regression (master seed: `2026111234`).
- **Validation & sampling sensitivity:** 20 repeats of stratified 5-fold cross-validation with strictly fold-internal preprocessing and reference fitting, plus phase-averaged post-occlusion grid thinning (60 s and 90 s).

Running `python run_all.py` writes all derived summary tables (`.csv`), leave-one-out estimates, and publication figures (`Figures 1–5` and `Figure S1` in vector `.pdf` and 600-dpi `.png`) to `results/`.

## License

MIT for source code. The cohort retains its original attribution and license.
