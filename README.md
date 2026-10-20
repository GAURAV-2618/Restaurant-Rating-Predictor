# Restaurant Rating Predictor 🍽️

## Overview

The Restaurant Rating Predictor is a machine learning project that predicts restaurant ratings based on restaurant-related features. It provides a Streamlit web interface where users can enter restaurant details and view predicted ratings.

## Features

- Predict restaurant ratings using machine learning.
- Preprocess restaurant data for model training.
- Compare multiple regression models.
- Evaluate model performance using MAE, RMSE, and R².
- Use a saved model artifact for predictions.
- Explore the dataset and model evaluation in a Jupyter Notebook.

## Tech Stack

- **Language:** Python
- **Web application:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Model persistence:** Joblib
- **Notebook:** Jupyter

## Project Structure

```text
Restaurant-Rating-Predictor/
├── app.py
├── model_utils.py
├── restaurant_predict.ipynb
├── Dataset.csv
├── requirements.txt
└── model_artifacts/
    └── restaurant_model.joblib
```

## Requirements

- Python 3.10 or 3.11 recommended
- pip
- The project files and trained model artifact

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/GAURAV-2618/Restaurant-Rating-Predictor.git
   ```

2. Enter the project directory:

   ```bash
   cd Restaurant-Rating-Predictor
   ```

3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate it on Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

## Model Evaluation

The project compares regression models using cross-validation and holdout evaluation. Metrics include:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination

The selected model and evaluation results are determined by the project's training workflow.

## Jupyter Notebook

Open `restaurant_predict.ipynb` in VS Code or Jupyter Notebook to explore the dataset, preprocessing, model training, and evaluation.

## Notes

- Keep `model_artifacts/restaurant_model.joblib` in the expected location so the application can load the trained model.
- Ensure `Dataset.csv` is available at the path expected by the application.
- Do not commit virtual environments, private credentials, or API keys.

