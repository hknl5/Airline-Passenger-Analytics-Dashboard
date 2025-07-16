# Airline Passenger Duty-Free Prediction System

This project builds a machine learning pipeline that predicts whether airline passengers will purchase from duty-free stores, estimates their spending, and provides product recommendations. It includes an interactive dashboard built with Streamlit for real-time prediction and analysis.

## Features

- Classification model to predict if a passenger will buy from the duty-free store
- Regression model to estimate the amount they are likely to spend
- Rule-based product recommendation system
- Interactive dashboard using Streamlit
- Feature engineering including categorical encoding and interaction terms
- Visual insights and performance metrics

## Project Structure

```
airline-ml-system/
├── data/
│   └── ariline_passenger.csv
├── models/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── label_encoders.pkl
│   └── feature_columns.pkl
├── pipeline/
│   └── pipeline.py
├── train_models.py
├── recommend.py
├── app.py
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/hknl5/airline-ml-system.git
cd airline-ml-system
```

### 2. Set up the environment

```bash
conda create -n airline-ml python=3.8
conda activate airline-ml
pip install -r requirements.txt
```

### 3. Train the models

```bash
python train_models.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- Streamlit

## Possible Improvements

- Replace Random Forest with Gradient Boosting models (e.g., XGBoost, CatBoost)
- Integrate collaborative filtering using purchase history
- Add explainability using SHAP values
- Deploy as an API using FastAPI or Flask

## License

This project is licensed under the MIT License.
