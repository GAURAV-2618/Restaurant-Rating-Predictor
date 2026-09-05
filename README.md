# Restaurant Rating Predictor

This is a simple Machine Learning project that predicts the rating of a restaurant from **0 to 5**.

The project is made using **Python, Machine Learning, and Streamlit**.

## What This Project Does

The app takes some restaurant information and predicts its rating.

You can enter:

* Average cost for two people
* Price range
* Number of votes
* Country
* City
* Cuisines
* Table booking
* Online delivery

The app then gives a predicted restaurant rating.

## Main Sections

The Streamlit app has 4 sections:

### 1. Prediction

Enter restaurant details and get the predicted rating.

### 2. Model Comparison

Compare the three Machine Learning models used in the project.

### 3. Dataset Overview

View the dataset and basic information about it.

### 4. Exploratory Analysis

View different graphs and charts from the dataset.

## Machine Learning Models

Three models are used:

* Linear Regression
* Decision Tree
* Random Forest

The models are trained using an **80/20 train-test split**.

The model with the best R² score is selected as the best model.

## Project Files

```text
restaurant-rating-predictor/
│
├── app.py
├── Dataset.csv
├── requirements.txt
├── Readme.md
└── restaurant_predict.ipynb
```

### File Description

* `app.py` - Main Streamlit application
* `Dataset.csv` - Restaurant dataset
* `requirements.txt` - Required Python libraries
* `Readme.md` - Project information
* `restaurant_predict.ipynb` - Jupyter Notebook

## How to Install

First, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Then install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Make sure `Dataset.csv` and `app.py` are in the same folder.

Run:

```bash
streamlit run app.py
```

The app will open in your browser.

If it does not open automatically, go to:

```text
http://localhost:8501
```

## Dataset

The dataset contains information about restaurants, such as:

* Restaurant name
* Country
* City
* Cuisines
* Average cost
* Price range
* Table booking
* Online delivery
* Votes
* Restaurant rating

The **Aggregate rating** column is used as the value that the model predicts.

## Data Cleaning

Before using the data:

* Restaurants with a rating of 0 are removed.
* Missing cuisine values are replaced with `Unknown`.
* Yes/No values are changed to numbers.
* The number of cuisines is calculated.
* City and country values are converted into numbers.

## Model Evaluation

The models are compared using:

* **MAE** - Shows the average error.
* **RMSE** - Shows the prediction error and gives more importance to large errors.
* **R² Score** - Shows how well the model predicts the rating.

For MAE and RMSE, a **lower value is better**.

For R², a **higher value is better**.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## Goal of the Project

The main goal of this project is to use Machine Learning to **predict restaurant ratings** based on restaurant information.
