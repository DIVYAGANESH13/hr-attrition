import pickle
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    result = ""  # Initialize result here to avoid UnboundLocalError
    error_message = ""

    if request.method == 'POST':
        try:
            # Retrieve form data for numerical features
            age = float(request.form['age'])
            income = float(request.form['income'])
            distance = float(request.form['distance'])
            education = float(request.form['education'])
            years_at_company = float(request.form['years_at_company'])
            job_satisfaction = float(request.form['job_satisfaction'])
            num_companies_worked = float(request.form['num_companies_worked'])

            # Prepare the input data for prediction
            input_data = [[age, income, distance, education, years_at_company, job_satisfaction, num_companies_worked]]

            # Use the model to predict
            prediction = model_pipeline.predict(input_data)[0]

            # Map the prediction to a result (Attrition or No Attrition)
            result = 'Attrition' if prediction == 1 else 'No Attrition'

        except Exception as e:
            error_message = f"Error occurred: {str(e)}"

    return render_template('index.html', prediction=result, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
