import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from recommend import recommend_products, get_all_product_categories
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Airline Passenger Analytics Dashboard",
    page_icon="✈️",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load the dataset"""
    df = pd.read_csv("ariline_passenger.csv")

    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Estimated_Income'] = pd.to_numeric(df['Estimated_Income'], errors='coerce')
    df['Flight_Duration_Hours'] = pd.to_numeric(df['Flight_Duration_Hours'], errors='coerce')
    df['Amount_Spent'] = pd.to_numeric(df['Amount_Spent'], errors='coerce')

    # Safely convert Bought_From_Duty_Free to 0/1
    if df['Bought_From_Duty_Free'].dtype == bool:
        df['Bought_From_Duty_Free'] = df['Bought_From_Duty_Free'].astype(int)
    else:
        df['Bought_From_Duty_Free'] = df['Bought_From_Duty_Free'].map({'True': 1, 'False': 0})

    # Keep only rows with complete data for essential fields
    df = df.dropna(subset=[
        'Age', 'Estimated_Income', 'Flight_Duration_Hours',
        'Amount_Spent', 'Bought_From_Duty_Free'
    ])

    return df


@st.cache_resource
def load_models():
    """Load trained models and encoders"""
    try:
        classifier = joblib.load('models/classifier.pkl')
        regressor = joblib.load('models/regressor.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return classifier, regressor, label_encoders, feature_columns
    except FileNotFoundError:
        st.error("Models not found. Please run train_models.py first.")
        return None, None, None, None

def prepare_input_features(input_data, label_encoders, feature_columns):
    """Prepare input data for prediction"""
    # Create feature vector
    features = {}
    
    # Numerical features
    features['Age'] = input_data['age']
    features['Estimated_Income'] = input_data['income']
    features['Flight_Duration_Hours'] = input_data['flight_duration']
    
    # Categorical features
    categorical_mappings = {
        'Gender': input_data['gender'],
        'Traveler_Type': input_data['traveler_type'],
        'Travel_Purpose': input_data['travel_purpose'],
        'Trip_Type': input_data['trip_type'],
        'Ticket_Class': input_data['ticket_class'],
        'Airline': input_data['airline']
    }
    
    for col, value in categorical_mappings.items():
        if col in label_encoders:
            try:
                features[col] = label_encoders[col].transform([value])[0]
            except ValueError:
                # Handle unseen categories
                features[col] = 0
    
    # Create DataFrame with correct column order
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.reindex(columns=feature_columns, fill_value=0)
    
    return feature_df

def main():
    st.title("Airline Passenger Analytics Dashboard")
    st.markdown("Predict passenger behavior and get product recommendations")
    
    # Load data and models
    df = load_data()
    classifier, regressor, label_encoders, feature_columns = load_models()
    
    if classifier is None:
        st.stop()
    
    # Sidebar for user input
    st.sidebar.header("Passenger Information")
    
    # Input fields
    age = st.sidebar.slider("Age", min_value=18, max_value=80, value=35)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    traveler_type = st.sidebar.selectbox("Traveler Type", ["Member", "Non-Member"])
    travel_purpose = st.sidebar.selectbox(
        "Travel Purpose", 
        ["Business", "Leisure", "Medical", "Education", "Family"]
    )
    income = st.sidebar.number_input(
        "Estimated Income", 
        min_value=10000, 
        max_value=200000, 
        value=50000
    )
    trip_type = st.sidebar.selectbox("Trip Type", ["Domestic", "International"])
    ticket_class = st.sidebar.selectbox("Ticket Class", ["Economy", "Business", "First"])
    flight_duration = st.sidebar.slider(
        "Flight Duration (Hours)", 
        min_value=1.0, 
        max_value=15.0, 
        value=5.0
    )
    airline = st.sidebar.selectbox(
        "Airline", 
        ["American Airlines", "Delta", "United", "Southwest", "JetBlue", "Other"]
    )
    
    # Prepare input data
    input_data = {
        'age': age,
        'gender': gender,
        'traveler_type': traveler_type,
        'travel_purpose': travel_purpose,
        'income': income,
        'trip_type': trip_type,
        'ticket_class': ticket_class,
        'flight_duration': flight_duration,
        'airline': airline
    }
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predictions")
        
        if st.button("Generate Predictions", type="primary"):
            # Prepare features
            features = prepare_input_features(input_data, label_encoders, feature_columns)
            
            # Make predictions
            duty_free_prob = classifier.predict_proba(features)[0]
            will_buy = classifier.predict(features)[0]
            spending_amount = regressor.predict(features)[0]
            
            # Display predictions
            st.metric("Will Buy Duty Free", "Yes" if will_buy else "No")
            st.metric("Purchase Probability", f"{duty_free_prob[1]:.2%}")
            st.metric("Estimated Spending", f"${spending_amount:.2f}")
            
            # Product recommendations
            recommendations = recommend_products(age, travel_purpose, ticket_class)
            st.subheader("Recommended Products")
            for i, product in enumerate(recommendations, 1):
                st.write(f"{i}. {product}")
    
    with col2:
        st.subheader("Dataset Overview")
        st.metric("Total Passengers", len(df))
        st.metric("Duty Free Buyers", df['Bought_From_Duty_Free'].sum())
        st.metric("Average Spending", f"${df['Amount_Spent'].mean():.2f}")
    
    # Visualizations
    st.header("Data Insights")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Spending Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        spending_data = df[df['Amount_Spent'] > 0]['Amount_Spent']
        ax.hist(spending_data, bins=30, alpha=0.7, color='skyblue')
        ax.set_xlabel('Amount Spent ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Spending Amount')
        st.pyplot(fig)
    
    with col4:
        st.subheader("Top 5 Products")
        product_counts = df['Product'].value_counts().head()

        if not product_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            product_counts.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_xlabel('Product Category')
            ax.set_ylabel('Purchase Count')
            ax.set_title('Most Popular Products')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No product purchase data available to display.")

    
    # Additional insights
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Travel Purpose of Buyers")
        buyers_df = df[df['Bought_From_Duty_Free'] == 1]
        purpose_counts = buyers_df['Travel_Purpose'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        purpose_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_ylabel('')
        ax.set_title('Travel Purpose Distribution (Buyers Only)')
        st.pyplot(fig)
    
    with col6:
        st.subheader("Spending by Ticket Class")
        class_spending = df.groupby('Ticket_Class')['Amount_Spent'].mean()
        fig, ax = plt.subplots(figsize=(8, 6))
        class_spending.plot(kind='bar', ax=ax, color='gold')


        ax.set_xlabel('Ticket Class')
        ax.set_ylabel('Average Spending ($)')
        ax.set_title('Average Spending by Ticket Class')
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    # Data table
    st.header("Sample Data")
    st.dataframe(df.head(10))

if __name__ == "__main__":
    main()
