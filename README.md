```
cancer_stats_project/
├── config/
│   └── data_paths.yaml          # central config for file locations
│
├── data/
│   ├── raw/                     # original CSVs
│   │   └── CRUK_EDHub_IncidenceByStageRCRD_DataTable2025-09-28.csv
│   └── processed/               # cleaned / transformed datasets
│
├── output/
│   ├── figures/                 # generated plots / figures
│   └── tables/                  # exported summary tables
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # main script orchestrating workflow
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py         # functions to load raw/processed CSVs
│   │   └── preprocess.py        # data cleaning and preprocessing
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py    # feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py       # model training
│   │   ├── evaluate_model.py    # model evaluation
│   │   └── model_utils.py       # helper functions for models
│   │
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py         # plotting functions
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_dev.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_load_data.py
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_train_model.py
│   └── test_evaluate_model.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── pyrightconfig.json


## Table of Contents
- [Overview](#overview)
- [Data Source](#data-source)
- [Installation](#installation)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations and Considerations](#limitations-and-considerations)
- [Conclusions](#conclusions)


## Overview

This project uses cancer incidence data from [Cancer Research UK](https://www.cancerresearchuk.org/) to visualize patterns between cancer sites and diagnosis stages from 2018 to 2025. The analysis is conducted in a Jupyter Notebook, accompanied by a Python script generated using the Jupytext extension.

After an ETL (Extract, Transform, Load) process, the data revealed distributions across cancer sites and stages, which were explored through visualizations using Matplotlib and Seaborn.

An unsupervised machine learning model — KMeans clustering — was applied to data that I preprocessed to remove nulls, normalized by feature engineering incidence proportions and reshaping to long form. Model performance was evaluated using the silhouette score, and dimensionality reduction was performed using Principal Component Analysis (PCA) to project the results into two dimensions for visualization.


## Data Source

Data obtained from [Cancer Research UK](https://www.cancerresearchuk.org/) — Incidence by Stage dataset (2018–2025).

## Installation

Clone the repo and install dependencies:

git clone https://github.com/yourusername/cancer_stats_project.git
cd cancer_stats_project
python -m venv cancer_stats_env
.\cancer_stats_env\Scripts\activate
pip install -r requirements.txt

Launch Jupyter Notebook
jupyter notebook cancer_stats.ipynb

## Methodology

- **ETL**: Cleaned and reshaped CSV data into long-form for analysis.
- **Visualization**: Used Matplotlib and Seaborn to explore distributions.
- **Clustering**: Applied KMeans clustering to proportions by cancer site/stage.
- **Dimensionality Reduction**: Used PCA to reduce features to 2D for visualization.
- **Evaluation**: Assessed clustering performance using the silhouette score.

## Results & Conclusions

### Clustering Performance

- The KMeans clustering model showed **good separation** between clusters in 2D space (after PCA), with **minimal overlap**.
- Cluster groupings appeared to **align with staging distributions** visualized earlier using bar plots, suggesting biological or clinical relevance.

---

### Cluster Interpretations

**🔹 Cluster 0**  
*Cancers: Gynaecological, Uterine, Cervical*  
- Female reproductive system cancers 
- **Observation:** Very high incidence of **stage 1**, with relatively few cases at other stages.  
  ➤ *This may indicate good screening program.*

---

**🔹 Cluster 1**  
*Cancers: Blood, Lung, Non-Hodgkin Lymphoma*  
- Systemic or hematological cancers, not confined to a single organ.  
- **Observation:** Dominated by **stage 4** incidence.  
  ➤ *Suggests late diagnosis or rapid progression.*

---

**🔹 Cluster 2**  
*Cancers: Kidney, Ovarian, Prostate*  
- Urological and reproductive system cancers.  
- **Observation:** High incidence at **stage 1**, but unusually low incidence at **stage 2**.  
  ➤ *May indicate difficulty distinguishing between early stages clinically.*

---

**🔹 Cluster 3**  
*Cancers: Bladder, Breast, Melanoma (Skin)*  
- Diverse set; harder to define biologically.  
- **Observation:** High incidence at **stages 1 and 2**, lower at 3 and 4.  
  ➤ *Could reflect better public awareness or effective screening programs.*

---

**🔹 Cluster 4**  
*Cancers: Bowel, Hodgkin Lymphoma*  
- Involves digestive and lymphatic systems.  
- **Observation:** Higher incidence at **later stages** compared to stage 1.  
  ➤ *Potential need for earlier detection tools or improved screening.*

---

### Insights from Cluster Analysis

- Cancers within the **same cluster may share underlying biology**, indicating potential for:
  - **Biomarker discovery**
  - **Shared therapeutic strategies**
  - **Patient stratification in clinical trials**
  - **Improved staging tools tailored to similar cancer types**

---

### Limitations & Further Considerations

- A subset of cancers (e.g. stomach, oesophageal-gastric, pancreatic) appear to **form a subcluster** between clusters 1 and 4, both of which show high stage 4 incidence. These digestive system cancers might warrant **a distinct cluster**.
  
- The PCA dimensionality reduction to 2D may have **oversimplified** the structure of the data:- Some cancers (e.g. bowel) may belong to a distinct other cluster.
- KMeans clustering assumes **spherical clusters**, which may not reflect true biological complexity.


### Conclusion

The clustering model uncovered patterns in cancer staging that appear biologically meaningful and clinically relevant. While the results are promising, further refinement — such as alternative clustering methods or higher-dimensional analysis — could improve classification and reveal deeper insights.
