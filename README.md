# MTG Preconstructed Deck Popularity & Recommender Pipeline

A complete end-to-end analytics and recommendation system for evaluating and improving preconstructed Commander decks. This project ingests raw deck data, cleans and enriches it, performs exploratory analysis, trains ensemble models to predict deck popularity, and constructs a hybrid recommender to suggest targeted card upgrades.

---

## Repository Structure

```
.
├── README.md                             # This overview
├── requirements.txt                      # Python dependencies
├── data/
│   ├── All_Precons_2025-05-18.csv        # Original raw export
│   └── Cleaned_Precon_Deck_Data.csv      # Cleaned, analysis-ready dataset
├── notebooks/
│   ├── EDA.ipynb                         # Exploratory Data Analysis with narrative
│   └── model_development.ipynb          # Model training, tuning, and explainability
├── report_figures/                       # Generated plots and tables (PNGs, CSVs)
├── src/
│   └── mtg_pipeline.py                   # Main CLI pipeline (data → model → recs)
├── docs/
│   ├── Report.html                       # Compiled technical report (HTML)
│   └── Report.pdf                        # Compiled technical report (PDF)
├── data_analysis/
│   ├── EDA.ipynb                        #Python script to generate visuals   
```

---

## Prerequisites

- **Python 3.8+**  
- **pip** (or **conda**) for package management  

---

## Data

- **Raw dataset**: `data/All_Precons_2025-05-18.csv`  
- **Cleaned dataset**: `data/Cleaned_Precon_Deck_Data.csv`  
  - Tokens/emblems removed  
  - Numeric fields coerced and imputed  
  - Date fields parsed and recency computed  
  - Feature-engineered columns added (e.g., `days_since_release`, `keyword_count`, `is_high_cmc`)

---

## Usage

### 1. Command-Line Pipeline

Run the full analysis & recommendation pipeline:

```bash
python src/mtg_pipeline.py \
  --data-path        data/Cleaned_Precon_Deck_Data.csv \
  --interactions-path data/interactions.csv \
  --model-out        out/model/xgb_model.json \
  --shap-out         report_figures/shap/ \
  --rec-out          report_figures/recs/user1.json
```

Use `--help` to see all flags:

```bash
python src/mtg_pipeline.py --help
```

Outputs:

- Trained models (XGBoost, RF, CatBoost) in JSON/pickle  
- SHAP & LIME explainability plots in `report_figures/shap/`  
- Cohort & trend visualizations in `report_figures/`  
- Top-K upgrade recommendations in JSON  

### 2. Jupyter Notebooks

- **`notebooks/EDA.ipynb`**  
  - Data cleaning, descriptive statistics, and exploratory plots  
- **`notebooks/Model_Development.ipynb`**  
  - Model training, hyperparameter tuning (Optuna), SHAP/LIME explainability, and evaluation  

Launch via:

```bash
jupyter lab notebooks/EDA.ipynb
jupyter lab notebooks/model_development.ipynb
```

---

## Reports

The compiled technical reports are available in `docs/`:

It contains all literature review, methodology, visualizations, and findings in publication-quality format.

---