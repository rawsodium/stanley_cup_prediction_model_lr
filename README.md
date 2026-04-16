## Project Structure

stanley-cup-prediction-model/
│
├── scripts/
│   ├── run_pipeline.py              # Runs current-season predictions + simulation
│   ├── evaluate_historical.py       # Backtests model on past seasons
│   ├── clean_historical_data.py     # One-time data cleaning script
│
├── src/
│   ├── features/
│   │   ├── build_features.py
│   │   ├── build_full_dataset.py
│   │
│   ├── models/
│   │   ├── logistic_regression_model.py
│   │
│   ├── processing/
│   │   ├── standings.py
│   │   ├── matchups.py
│   │   ├── history.py
│   │   ├── team_stats.py
│   │   ├── advanced_stats.py
│   │   ├── build_training_dataset.py
│   │
│   ├── utils/
│   │   ├── cache.py
│
├── cleaned_data/
├── outputs/
├── requirements.txt
└── README.md

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Current Season pipeline
```bash
python -m scripts.run_pipeline
```

Outputs are saved in outputs/

### 3. Run Historical Evaluation from 2021-2025
```bash
python -m scripts.evaluate_historical
```