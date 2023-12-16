# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
consolidated_data = pd.read_csv("consolidated_data.csv")

# Display the top 10 satisfied customers
st.subheader("Top 10 Satisfied Customers")
st.table(consolidated_data[['MSISDN/Number', 'satisfaction_score']].sort_values(by='satisfaction_score', ascending=False).head(10))

# Visualize predicted vs. actual values
st.subheader("Actual vs. Predicted Satisfaction Score")
fig, ax = plt.subplots()
ax.scatter(consolidated_data['satisfaction_score'], consolidated_data['predicted_satisfaction_score'])
ax.set_xlabel('Actual Satisfaction Score')
ax.set_ylabel('Predicted Satisfaction Score')
st.pyplot(fig)

# Pie chart: Percentage Distribution of Satisfaction Levels
st.subheader("Percentage Distribution of Satisfaction Levels")
satisfaction_counts = consolidated_data['Satisfaction Level'].value_counts()
fig_pie = px.pie(satisfaction_counts, names=satisfaction_counts.index, values=satisfaction_counts.values)
st.plotly_chart(fig_pie)

# Scatter plots using Plotly Express
fig1 = px.scatter(consolidated_data, x='satisfaction_score', y='Engagement Score', title='Satisfaction Score vs Engagement Score')
fig2 = px.scatter(consolidated_data, x='satisfaction_score', y='experience_score', title='Satisfaction Score vs Experience Score')
fig3 = px.scatter(consolidated_data, x='Engagement Score', y='experience_score', title='Engagement Score vs Experience Score')

# Show the figures
st.subheader("Scatter Plots")
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

# Header and instructions
st.title("🌟 Predictive Model for Satisfaction Score 🌟")
st.write("Input your experience and engagement scores to predict satisfaction.")

# Get the maximum values for experience_score and engagement_score from the DataFrame
max_experience_score = consolidated_data['experience_score'].max()
max_engagement_score = consolidated_data['Engagement Score'].max()

# User input for experience and engagement scores
experience_score = st.slider("😊 Experience Score", min_value=0.0, max_value=max_experience_score, step=0.01, value=0.5)
engagement_score = st.slider("💼 Engagement Score", min_value=0.0, max_value=max_engagement_score, step=0.01, value=0.5)

# Display the user-inputted scores
st.write(f"Experience Score: {experience_score:.2f}")
st.write(f"Engagement Score: {engagement_score:.2f}")

# Prepare the input for prediction
user_input = pd.DataFrame({'experience_score': [experience_score], 'Engagement Score': [engagement_score]})

# Ensure the order of features in user_input matches the order during training
user_input = user_input[['Engagement Score', 'experience_score']]

# Select features for the regression model
regression_features = ['Engagement Score', 'experience_score']

# Drop rows with missing values in the selected features
regression_data = consolidated_data.dropna(subset=regression_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    regression_data[regression_features],
    regression_data['satisfaction_score'],
    test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Ridge regression model with regularization parameter alpha
model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)

# Train the model on the training set
model.fit(X_train_scaled, y_train)

# Make predictions for the user input
user_input_scaled = scaler.transform(user_input)
predicted_satisfaction = model.predict(user_input_scaled)

# Display the predicted satisfaction score
st.subheader("🔮 Predicted Satisfaction Score:")
st.write(predicted_satisfaction[0])

# Map satisfaction levels based on predicted satisfaction score
# Define thresholds for satisfaction categories relative to the highest score
max_satisfaction_score = consolidated_data['satisfaction_score'].max()

low_satisfaction_threshold = 0.25 * max_satisfaction_score
moderate_satisfaction_threshold = 0.50 * max_satisfaction_score
high_satisfaction_threshold = 0.75 * max_satisfaction_score

# Map satisfaction levels based on predicted satisfaction score
def map_satisfaction_level(score):
    if score < low_satisfaction_threshold:
        return "Low"
    elif low_satisfaction_threshold <= score < moderate_satisfaction_threshold:
        return "Moderate"
    elif moderate_satisfaction_threshold <= score < high_satisfaction_threshold:
        return "Satisfied"
    else:
        return "High"

# Map satisfaction level
satisfaction_level = map_satisfaction_level(predicted_satisfaction[0])
st.subheader("😃 Satisfaction Level:")
st.write(satisfaction_level)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the model performance metrics
st.subheader("📊 Model Performance Metrics:")
st.write(f'Mean Squared Error: {mse}')
st.write(f'R-squared: {r2}')

# Add a footer or contact section
st.markdown("---")
st.subheader("🙋 Need Help or Have Feedback?")
st.write("Contact us at support@example.com")

# Add some style to the layout
st.markdown('<style>body {background-color: #f2f2f2;}</style>', unsafe_allow_html=True)