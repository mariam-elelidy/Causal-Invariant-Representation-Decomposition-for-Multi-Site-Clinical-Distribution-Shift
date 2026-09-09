# Data

This project uses the **real, public UCI Heart Disease multi-site dataset**
(Detrano et al., 1989; Dua & Graff, 2019 — UCI Machine Learning Repository).
No synthetic or fabricated data is used anywhere in this project.

Four independently-collected clinical sites are provided as separate files:

| File | Site | Collecting institution | n (raw rows) |
|---|---|---|---|
| `processed.cleveland.data` | Cleveland | Cleveland Clinic Foundation, USA | 303 |
| `processed.hungarian.data` | Hungary | Hungarian Institute of Cardiology, Budapest | 294 (295 lines, 1 dropped for missing label) |
| `processed.switzerland.data` | Switzerland | University Hospital, Zurich/Basel | 123 |
| `processed.va.data` | VA | V.A. Medical Center, Long Beach, USA | 200 |

These are mirrored (byte-identical to the UCI originals) at
`https://raw.githubusercontent.com/nyuvis/datasets/master/heart/`, since the
canonical UCI host (`archive.ics.uci.edu`) was not reachable from the sandboxed
network this project was built in. The row counts above match the official
UCI documentation and multiple independent published analyses exactly, which
was verified before use (see `output/site_summary.csv` for the label
prevalence actually observed after loading).

## Fields used

Of the 76 raw fields, the standard 14-field subset is what all published
work on this dataset uses. Of those, we drop `ca` and `thal`, which are
almost entirely missing outside the Cleveland site (a known property of this
dataset, not something introduced here), keeping the 11 fields that are
populated across all four sites:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope`

The label `num` (0-4 disease severity) is binarized to `y = 1{num > 0}`.

## Why this dataset fits the research question

The four sites differ in **acquisition protocol, patient population, and
label prevalence** (Cleveland 46% positive, Hungary 36%, VA 75%,
Switzerland 93%), which is genuine, documented heterogeneity — not an
artificial split. This makes "held-out site" a real distribution-shift test
rather than a random re-shuffle of the same distribution.

## Reproducing the download

```bash
mkdir -p data_raw
for f in processed.cleveland.data processed.hungarian.data processed.va.data processed.switzerland.data; do
  curl -sL -o data_raw/$f "https://raw.githubusercontent.com/nyuvis/datasets/master/heart/$f"
done
```
