# AI_Predictive_System/app/app.py

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('./model/heart_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: heart_model.pkl not found. Please ensure it's in the 'model/' directory.")
    print("Run model_training.ipynb to train and save the model.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the expected columns for the model input (based on training data)
feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # --- START DEBUGGING BLOCK ---
            print("\n--- Incoming Form Data ---")
            for key, value in request.form.items():
                print(f"{key}: '{value}' (Type: {type(value)})")
            print("--------------------------\n")
            # --- END DEBUGGING BLOCK ---

            # Get form data
            # Use .get() with a default value or explicit check to handle missing fields gracefully
            # or try-except around individual conversions.
            # For debugging, we will keep current structure but add more specific error handling.
            
            age_str = request.form.get('age')
            sex_str = request.form.get('sex')
            cp_str = request.form.get('cp')
            trestbps_str = request.form.get('trestbps')
            chol_str = request.form.get('chol')
            fbs_str = request.form.get('fbs')
            restecg_str = request.form.get('restecg')
            thalch_str = request.form.get('thalch') # Corrected name
            exang_str = request.form.get('exang')
            oldpeak_str = request.form.get('oldpeak')
            slope_str = request.form.get('slope')
            ca_str = request.form.get('ca')
            thal_str = request.form.get('thal')

            # Convert to appropriate types with more specific error handling
            try:
                age = int(age_str)
                sex = int(sex_str)
                cp = int(cp_str)
                trestbps = int(trestbps_str)
                chol = int(chol_str)
                fbs = int(fbs_str)
                restecg = int(restecg_str)
                thalch = int(thalch_str) # Corrected name
                exang = int(exang_str)
                oldpeak = float(oldpeak_str)
                slope = int(slope_str)
                ca = int(ca_str)
                thal = int(thal_str)
            except ValueError as ve:
                error_message = f"Data conversion error: {ve}. Please check your inputs. Problem with value for: "
                # Attempt to pinpoint the exact problematic field
                if not age_str or not age_str.isdigit(): error_message += "Age "
                if not sex_str or not sex_str.isdigit(): error_message += "Sex "
                if not cp_str or not cp_str.isdigit(): error_message += "Chest Pain Type "
                if not trestbps_str or not trestbps_str.isdigit(): error_message += "Resting Blood Pressure "
                if not chol_str or not chol_str.isdigit(): error_message += "Cholesterol "
                if not fbs_str or not fbs_str.isdigit(): error_message += "Fasting Blood Sugar "
                if not restecg_str or not restecg_str.isdigit(): error_message += "Resting ECG "
                if not thalch_str or not thalch_str.isdigit(): error_message += "Max Heart Rate " # Corrected name
                if not exang_str or not exang_str.isdigit(): error_message += "Exercise Angina "
                try: float(oldpeak_str) # Test oldpeak separately as it's float
                except ValueError: error_message += "Oldpeak "
                if not slope_str or not slope_str.isdigit(): error_message += "Slope "
                if not ca_str or not ca_str.isdigit(): error_message += "CA "
                if not thal_str or not thal_str.isdigit(): error_message += "Thal "
                
                print(f"Flask APP Error (ValueError): {error_message}")
                return render_template('result.html', result=f"Error processing input: {error_message}")
            except TypeError as te:
                 print(f"Flask APP Error (TypeError): {te}. One of the form fields might be missing or None.")
                 return render_template('result.html', result=f"Error processing input: Missing form field. {te}")
            
            # Create a DataFrame from the input data, maintaining column order
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]],
                                      columns=feature_columns)

            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            result_text = "You are likely to have Heart Disease." if prediction == 1 else "You are likely NOT to have Heart Disease."
            prob_no_disease = f"{prediction_proba[0]*100:.2f}%"
            prob_disease = f"{prediction_proba[1]*100:.2f}%"

            return render_template('result.html',
                                   result=result_text,
                                   prob_no_disease=prob_no_disease,
                                   prob_disease=prob_disease)

        except Exception as e:
            print(f"Flask APP General Error: {e}")
            return render_template('result.html', result=f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    if 'model' in locals() and model is not None:
        app.run(debug=True)
    else:
        print("Flask app not started because the model could not be loaded.")