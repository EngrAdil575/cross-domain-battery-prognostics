# Cross-Domain Transfer Learning for Lithium-Ion Battery Health Prognostics: A Digital Twin Approach

Datasets, code, results, and the offline digital twin for the conference paper:

> **Cross-Domain Transfer Learning for Lithium-Ion Battery Health Prognostics: A Digital Twin Approach**
> Muhammad Waiz Wais, Roya Mangal, Muhammad Sadiq Orakzai, Adil Zohaib, Haris Sheh Zad, Sohail Khalid.

This repository is the reproducibility record for the paper. It holds the two datasets used in the study, the full analysis code, every result table and figure, and the offline digital twin.

## What the paper does

New battery types rarely come with the long aging records that deep prognostic models need, which delays digital-twin deployment. The paper addresses this with cross-domain transfer learning: a model trained on one battery type (source domain) is reused for a different type (target domain). The proposed **GRU-SDR** is a GRU pretrained on the source domain and fine-tuned with little target-domain data; it predicts State of Health (SOH) from early-discharge features, and derives Remaining Useful Life (RUL) from the filtered SOH (SOH-derived RUL) instead of learning it directly, with no extra trainable parameters.

All models are pretrained on four NASA cells, fine-tuned on the first 10–100% of the life of one CALCE CS2 cell with a known lifetime, and tested on three unseen CALCE cells without their capacity history. GRU-SDR is compared against six deep networks that predict SOH and RUL jointly. With 10% of the target data, fine-tuning lowers the RUL RMSE of the standard models by 29–39% and the SOH RMSE by up to 51%. GRU-SDR reaches an SOH RMSE of 2.2% and an RUL RMSE of 93 cycles (26% lower than the best directly learned RUL). An offline digital twin driven by GRU-SDR flags the warning and maintenance states of an unseen cell.

## Repository layout

```
cross-domain-battery-prognostics/
├── data/
│   ├── NASA/                     Source domain: NASA cells B0005, B0006, B0007, B0018 (.mat)
│   ├── CALCE_processed/          Target domain: per-cycle features for CALCE CS2_35–CS2_38 (.csv)
│   └── DATASET_README.txt        Sources, columns, and licensing of both datasets
├── code/
│   ├── notebooks/
│   │   ├── TL_NASA_to_CALCE.ipynb        Main transfer-learning experiment (Colab, GPU)
│   │   └── Battery_Digital_Twin.ipynb    Interactive offline digital twin
│   ├── src/                      The same code as plain scripts, plus the Fig. 1 workflow source
│   ├── requirements.txt
│   └── CODE_README.txt           File-by-file description
├── digital_twin/
│   └── Battery_Digital_Twin.html Self-contained offline digital twin (open in a browser)
├── results/
│   ├── figures/                  Figs 1–6 (PDF for LaTeX, PNG for viewing)
│   └── tables/                   Main results, all runs, per-cell predictions, and settings.json
├── CITATION.cff
└── LICENSE                       MIT (code). The datasets keep their original providers' terms.
```

## Datasets

Two public datasets are used, in `data/`:

1. **NASA Battery Data Set** (source domain) — cells B0005, B0006, B0007, B0018 as the original `.mat` files. B. Saha and K. Goebel, "Battery data set," NASA Ames Prognostics Center of Excellence (PCoE) Data Repository, 2007. U.S. Government work, public domain.
2. **CALCE CS2 dataset** (target domain) — per-cycle features for CS2_35 to CS2_38, extracted from the original CALCE `.xlsx` files by the notebook. Source: CALCE Battery Research Group, University of Maryland. The original `.xlsx` files are not redistributed here; download them from CALCE if you need the raw cycling data.

Columns and extraction details are in `data/DATASET_README.txt`. Both datasets remain subject to the terms of their original providers.

## Reproducing the results

Open `code/notebooks/TL_NASA_to_CALCE.ipynb` in Google Colab (GPU), put the `NASA` and `CALCE` folders in Google Drive, and Runtime → Run all. The notebook pretrains every model on the NASA cells, fine-tunes on 10/25/50/100% of one CALCE cell over three seeds (42, 7, 123), tests on the unseen CALCE cells, and writes the tables and figures. Then run `code/notebooks/Battery_Digital_Twin.ipynb` for the digital twin. `code/src/` holds the same code as plain scripts, and `make_figures.py` regenerates Figs 2–4 from the result tables.

Dependencies are in `code/requirements.txt`: tensorflow (≥2.12), numpy, pandas, scipy, scikit-learn, matplotlib, and openpyxl.

## Results

`results/figures/` holds Figures 1–6 and `results/tables/` holds the result tables: `Table_main_results.csv` (mean ± SD over three seeds per model, strategy, and data share), `Table_all_runs.csv` (every run), `Table_proposed_runs.csv` (GRU-SDR per seed and cell), the per-cycle prediction tables, and `settings.json`.

## Offline digital twin

`digital_twin/Battery_Digital_Twin.html` is a self-contained offline interface — open it in any browser, no server or internet needed. Driven by GRU-SDR, it reports the SOH, the RUL, and a filtered health status (healthy / warning / maintenance) for an unseen cell.

## Citation

If you use this code or data, please cite the conference paper (see `CITATION.cff`) and the two datasets (NASA PCoE, Saha and Goebel, 2007; CALCE Battery Research Group, University of Maryland).

## License

The code in this repository is released under the MIT License (see `LICENSE`). The datasets in `data/` keep their original providers' terms.
