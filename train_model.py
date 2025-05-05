import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Load data
data = pd.read_csv('dataset.csv')  # Ensure the correct path

# Clean up column names
data.columns = data.columns.str.strip()

# Feature selection (only numerical columns)
features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'Education', 'YearsAtCompany', 'JobSatisfaction', 'NumCompaniesWorked']
X = data[features]  # Only numerical features
y = data['Attrition']  # Target column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing pipeline with scaling and the model
model_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Standardize numerical features
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model_pipeline, model_file)

# Print accuracy score
print(f"Model accuracy: {model_pipeline.score(X_test, y_test)}")
