"""
 deploying developer salary prediction model using flask

to run:
python  app/flask_api.py

then open browser:
htty://127.0.0.1.:5000
"""

from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask,request,render_template
import joblib

# app setup
app = Flask(__name__,template_folder="templates")

# model_path
root_dir = Path(__file__).resolve().parent
MODEL_PATH = root_dir.parent/"model"/"salary_pipeline.pkl"
print(f"Loading model from:{MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

#html form




@app.route('/') # SET THE ROOT PATH i.ewhen one visits http://127.0.0.1:5000
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_df = pd.DataFrame([{
        'Country': request.form.get('country'),
        'YearsCode': float(request.form.get('years')),
        'EdLevel': request.form.get('education'),
        'Employment': request.form.get('employment'),
        'LanguageCount': float(request.form.get('languages'))
    }])
    prediction = pipeline.predict(input_df)[0]
    salary= f"{int(np.clip(prediction,10000,500000 )):,}"
    
    return render_template("index.html", salary= salary)


if __name__ == '__main__':
    app.run(debug=True)
