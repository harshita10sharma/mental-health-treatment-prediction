from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# -------------------------------
# Load Saved Model, Scaler, Columns
# -------------------------------
model = joblib.load('mental_health_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# -------------------------------
# Helper function for prediction
# -------------------------------
def mental_health_score(prob):
    score = int(prob * 100)
    if score < 40:
        return f"Score: {score} — You seem mentally healthy 🙂"
    elif score < 70:
        return f"Score: {score} — Moderate stress, stay mindful 🧠"
    else:
        return f"Score: {score} — High stress, consider seeking support 💬"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        Age = int(request.form['Age'])
        Gender = request.form['Gender']
        Country = request.form['Country']
        self_employed = request.form['self_employed']
        family_history = request.form['family_history']
        no_employees = request.form['no_employees']
        remote_work = request.form['remote_work']
        tech_company = request.form['tech_company']
        benefits = request.form['benefits']
        care_options = request.form['care_options']
        wellness_program = request.form['wellness_program']
        seek_help = request.form['seek_help']
        anonymity = request.form['anonymity']
        mental_health_consequence = request.form['mental_health_consequence']
        phys_health_consequence = request.form['phys_health_consequence']
        coworkers = request.form['coworkers']
        supervisor = request.form['supervisor']
        mental_health_interview = request.form['mental_health_interview']
        phys_health_interview = request.form['phys_health_interview']
        mental_vs_physical = request.form['mental_vs_physical']
        obs_consequence = request.form['obs_consequence']

        # Create DataFrame
        input_data = pd.DataFrame([[Age, Gender, Country, self_employed, family_history, 
                                   no_employees, remote_work, tech_company, benefits, 
                                   care_options, wellness_program, seek_help, anonymity, 
                                   mental_health_consequence, phys_health_consequence, 
                                   coworkers, supervisor, mental_health_interview, 
                                   phys_health_interview, mental_vs_physical, obs_consequence]],
                                   columns=['Age', 'Gender', 'Country', 'self_employed', 'family_history',
                                            'no_employees', 'remote_work', 'tech_company', 'benefits',
                                            'care_options', 'wellness_program', 'seek_help', 'anonymity',
                                            'mental_health_consequence', 'phys_health_consequence',
                                            'coworkers', 'supervisor', 'mental_health_interview',
                                            'phys_health_interview', 'mental_vs_physical', 'obs_consequence'])

        # One-hot encoding like training
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_encoded)

        # Predict
        proba = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]
        result = "Yes, treatment likely needed ❤️" if pred == 1 else "No, treatment may not be needed 🙂"
        score_text = mental_health_score(proba)

        return render_template('index.html', prediction=result, score=score_text)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
