## Project Structure

<img width="633" height="560" alt="image" src="https://github.com/user-attachments/assets/90087cbf-db84-42d3-8d83-76455b1e72bb" />


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
