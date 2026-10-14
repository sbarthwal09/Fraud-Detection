# Fraud Detection

A machine-learning project that detects fraudulent mobile-money transactions. It performs
exploratory data analysis and trains a scikit-learn pipeline, then serves the trained
model through a Streamlit web app that labels a single transaction as **Fraud** or **Safe**.

## Dataset

- **Source:** [Kaggle — Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)
  (PaySim-style synthetic mobile-money transactions).
- **Local file:** `AIML Dataset.csv` (~493 MB, 6,362,620 rows, 11 columns).
- Downloaded in the notebook via `kagglehub.dataset_download("amanalisiddiqui/fraud-detection-dataset")`.

| Column | Description |
|---|---|
| `step` | Simulated time unit (1–743) |
| `type` | Transaction type: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | Transaction amount |
| `nameOrig` / `nameDest` | Sender / receiver account IDs |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before / after |
| `oldbalanceDest` / `newbalanceDest` | Receiver balance before / after |
| `isFraud` | Target label (1 = fraud) |
| `isFlaggedFraud` | Legacy flag from the source dataset |

### Class balance

- `isFraud`: **8,213 fraud** out of 6,362,620 transactions (**~0.13%**), 0 missing values.
- Fraud only ever occurs in `TRANSFER` and `CASH_OUT`:
  - `TRANSFER`: 4,097 frauds (0.769%) · `CASH_OUT`: 4,116 frauds (0.184%)
  - `PAYMENT`, `CASH_IN`, `DEBIT`: 0 frauds

## Approach

1. **EDA** — class balance, transaction-type distributions, log-scale amount histogram,
   amount-vs-fraud boxplot, frauds over time (`step`), top accounts, and a correlation heatmap.
2. **Feature engineering** — `balanceDiffOrig` and `balanceDiffDest` are derived for EDA
   and dropped before modelling. `step`, `nameOrig`, `nameDest`, and `isFlaggedFraud`
   are also dropped, leaving 6 model inputs (`type` + 5 balances/amount).
3. **Preprocessing** — `ColumnTransformer`:
   - `StandardScaler` on the 5 numeric columns
     (`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)
   - `OneHotEncoder(drop="first")` on `type`
   - `remainder="drop"`
4. **Model** — `LogisticRegression(class_weight="balanced", max_iter=1000)` wrapped in a
   scikit-learn `Pipeline` (class weights handle the class imbalance).
5. **Evaluation** — 70/30 stratified train/test split (`random_state=42`);
   `classification_report` + `confusion_matrix`.
6. **Persistence** — `joblib.dump(...)` → `fraud_detection_pipeline.pkl`.


## Project structure

```
Fraud Detection/
├── analysis_model.ipynb            # EDA + training notebook (source of the model)
├── fraud_detection.py              # Streamlit prediction app (entry point)
├── fraud_detection_pipeline.pkl    # Trained scikit-learn pipeline (joblib)
├── AIML Dataset.csv                # Dataset (not committed)
├── requirements.txt                # Python dependencies
└── README.md
```

## Getting started

Requires Python 3.10+.

```powershell
pip install -r requirements.txt
```

### Train / analyze

```powershell
jupyter lab analysis_model.ipynb
```

Run the cells top to bottom. Cell 0 downloads the data with `kagglehub` (needs Kaggle
credentials); the later cells read the local `AIML Dataset.csv`, so Cell 0 can be skipped
if the CSV is already present. Running the notebook regenerates
`fraud_detection_pipeline.pkl`.

### Run the web app

```powershell
streamlit run fraud_detection.py
```

Enter the transaction type, amount, and sender/receiver balances, then click **Predict**.
