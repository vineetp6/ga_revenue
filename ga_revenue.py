import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Set Streamlit title and description
st.title("Google Analytics Customer Revenue Prediction")
st.write("""
This app uses the Google Analytics Customer Revenue Prediction dataset to predict customer revenue based on user sessions.
You can visualize key insights from the dataset and predict revenue for specific sessions.
""")

# Load Dataset
@st.cache
def load_data():
    # Replace this with the path to your dataset
    data = pd.read_csv("train_v2.csv")
    return data

# Load data
data = load_data()

# Show dataset preview
st.header("Dataset Preview")
st.write(data.head())

# Data Preprocessing
st.header("Data Preprocessing")
# Extracting relevant features and target
data['totals.transactionRevenue'] = data['totals.transactionRevenue'].fillna(0).astype(float)
data['totals.pageviews'] = data['totals.pageviews'].fillna(0).astype(float)

# Selecting features (simplified for the example)
features = ['totals.pageviews', 'totals.hits', 'totals.timeOnSite', 'totals.bounces']
target = 'totals.transactionRevenue'

# Subset the data
df = data[features + [target]].dropna()

# Splitting the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Importance Visualization
st.header("Feature Importance")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Plotting feature importance
importances = model_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices])
plt.tight_layout()
st.pyplot(plt)

# Model Training
st.header("Train the Model")
if st.button("Train Model"):
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

# Revenue Prediction
st.header("Predict Customer Revenue")
st.write("Use the features below to predict the customer revenue for a session.")

pageviews = st.number_input("Pageviews", min_value=0, max_value=100, value=1)
hits = st.number_input("Hits", min_value=0, max_value=100, value=1)
time_on_site = st.number_input("Time on Site (in seconds)", min_value=0, max_value=10000, value=1)
bounces = st.number_input("Bounces", min_value=0, max_value=100, value=0)

# Create input data
input_data = pd.DataFrame([[pageviews, hits, time_on_site, bounces]], columns=features)

# Make prediction
if st.button("Predict Revenue"):
    predicted_revenue = model_rf.predict(input_data)[0]
    st.write(f"Predicted Revenue: ${predicted_revenue:.2f}")

# Visualizations
st.header("Revenue Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(df['totals.transactionRevenue'], kde=True)
st.pyplot(plt)

st.header("Pageviews vs Revenue")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['totals.pageviews'], y=df['totals.transactionRevenue'])
plt.title("Pageviews vs Transaction Revenue")
st.pyplot(plt)

st.header("Hits vs Revenue")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['totals.hits'], y=df['totals.transactionRevenue'])
plt.title("Hits vs Transaction Revenue")
st.pyplot(plt)

st.header("Time on Site vs Revenue")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['totals.timeOnSite'], y=df['totals.transactionRevenue'])
plt.title("Time on Site vs Transaction Revenue")
st.pyplot(plt)
