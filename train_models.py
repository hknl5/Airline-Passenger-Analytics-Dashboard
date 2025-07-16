import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the airline passenger dataset"""
    # Load data
    df = pd.read_csv("ariline_passenger.csv")

    
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Convert data types
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Estimated_Income'] = pd.to_numeric(df['Estimated_Income'], errors='coerce')
    df['Flight_Duration_Hours'] = pd.to_numeric(df['Flight_Duration_Hours'], errors='coerce')
    df['Amount_Spent'] = pd.to_numeric(df['Amount_Spent'], errors='coerce')
    
    # Convert boolean target
    if df['Bought_From_Duty_Free'].dtype == 'bool':
        df['Bought_From_Duty_Free'] = df['Bought_From_Duty_Free'].astype(int)
    else:
        df['Bought_From_Duty_Free'] = df['Bought_From_Duty_Free'].map({'True': 1, 'False': 0})

    df = df.dropna(subset=['Age', 'Estimated_Income', 'Flight_Duration_Hours', 'Amount_Spent', 'Bought_From_Duty_Free'])

    
    print(f"After preprocessing: {df.shape}")
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    # Select features for modeling
    feature_columns = [
        'Age', 'Gender', 'Traveler_Type', 'Travel_Purpose', 
        'Estimated_Income', 'Trip_Type', 'Ticket_Class', 
        'Flight_Duration_Hours', 'Airline'
    ]
    
    # Create feature dataframe
    X = df[feature_columns].copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Gender', 'Traveler_Type', 'Travel_Purpose', 
                       'Trip_Type', 'Ticket_Class', 'Airline']
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    return X, label_encoders

def train_classification_model(X, y):
    """Train classification model to predict Bought_From_Duty_Free"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=10000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification Model Performance:")
    print(classification_report(y_test, y_pred))
    
    return clf

def train_regression_model(X, y):
    """Train regression model to predict Amount_Spent"""
    # Filter out zero spending for better model
    non_zero_mask = y > 0
    X_filtered = X[non_zero_mask]
    y_filtered = y[non_zero_mask]
    
    if len(X_filtered) == 0:
        print("No non-zero spending data found, using all data")
        X_filtered, y_filtered = X, y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Regressor
    reg = RandomForestRegressor(n_estimators=9000, random_state=42)
    reg.fit(X_train, y_train)
    
    # Evaluate
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Regression Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    return reg

def main():
    """Main training function"""
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Prepare features
    X, label_encoders = prepare_features(df)
    
    # Prepare targets
    y_classification = df['Bought_From_Duty_Free']
    y_regression = df['Amount_Spent']
    
    print(f"Features shape: {X.shape}")
    print(f"Classification target distribution:")
    print(y_classification.value_counts())
    print(f"Regression target stats:")
    print(y_regression.describe())
    
    # Train models
    print("\nTraining Classification Model...")
    classifier = train_classification_model(X, y_classification)
    
    print("\nTraining Regression Model...")
    regressor = train_regression_model(X, y_regression)
    
    # Save models
    joblib.dump(classifier, 'models/classifier.pkl')
    joblib.dump(regressor, 'models/regressor.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')
    
    print("\nModels saved successfully!")
    print("- models/classifier.pkl")
    print("- models/regressor.pkl")
    print("- models/label_encoders.pkl")
    print("- models/feature_columns.pkl")

if __name__ == "__main__":
    main()
