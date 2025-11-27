from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    print("Error: Model files not found. Run train_model.py first.")
    exit()

@app.route('/')
def home():
    try:
        domains = encoders['Interested Domain'].classes_
    except KeyError:
        domains = []
    return render_template('index.html', domains=domains)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form.get('Gender')
        age = float(request.form.get('Age'))
        
        cgpa = float(request.form.get('CGPA'))
        gpa = (cgpa / 10.0) * 4.0
        
        domain = request.form.get('Interested Domain')
        python = request.form.get('Python')
        sql = request.form.get('SQL')
        java = request.form.get('Java')

        skill_map = {'Weak': 1, 'Average': 2, 'Strong': 3}
        
        try:
            gender_encoded = encoders['Gender'].transform([gender])[0]
            domain_encoded = encoders['Interested Domain'].transform([domain])[0]
        except ValueError:
            gender_encoded = 0
            domain_encoded = 0

        input_data = [
            gender_encoded,
            age,
            gpa,
            domain_encoded,
            skill_map.get(python, 1),
            skill_map.get(sql, 1),
            skill_map.get(java, 1)
        ]

        final_input = np.array([input_data])
        prediction_index = model.predict(final_input)[0]
        
        predicted_career = encoders['Future Career'].inverse_transform([prediction_index])[0]

        return render_template('result.html', prediction=predicted_career)

if __name__ == '__main__':
    app.run(debug=True)