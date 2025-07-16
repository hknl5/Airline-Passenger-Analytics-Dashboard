import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the provided URL
df = pd.read_csv("ariline_passenger.csv")


print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst few rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nTarget Variables Analysis:")
print("\nBought_From_Duty_Free distribution:")
print(df['Bought_From_Duty_Free'].value_counts())

print("\nAmount_Spent statistics:")
print(df['Amount_Spent'].describe())

print("\nProduct categories:")
print(df['Product'].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# Check unique values for categorical columns
categorical_cols = ['Nationality', 'Gender', 'Traveler_Type', 'Travel_Purpose', 
                   'Trip_Type', 'Ticket_Class', 'Airline', 'Payment_Method']

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col} unique values: {df[col].nunique()}")
        print(df[col].value_counts().head())
