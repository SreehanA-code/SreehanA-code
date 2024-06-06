from flask import Flask, render_template, request
import pandas as pd
from model import load_and_preprocess_data, train_models, predict_outcome, suggest_resource_allocation, identify_potential_risks

app = Flask(__name__)

# Load and preprocess data, train models
file_path = 'historical_project_data.csv'  # Replace with actual file path
df, preprocessor = load_and_preprocess_data(file_path)
model = train_models(df, preprocessor)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        project_type = request.form['ProjectType']
        budget = float(request.form['Budget'])
        duration = float(request.form['Duration'])
        team_size = int(request.form['TeamSize'])
        tasks = int(request.form['Tasks'])
        complexity = int(request.form['Complexity'])
        
        new_project = pd.DataFrame([{
            'ProjectType': project_type,
            'Budget': budget,
            'Duration': duration,
            'TeamSize': team_size,
            'Tasks': tasks,
            'Complexity': complexity
        }])
        
        outcome_prediction = predict_outcome(model, new_project)
        resource_suggestion = suggest_resource_allocation(new_project.iloc[0])
        risks = identify_potential_risks(new_project.iloc[0])
        
        return render_template('index.html', 
                               prediction=outcome_prediction[0], 
                               allocation=resource_suggestion, 
                               risks=risks)

if __name__ == "__main__":
    app.run(debug=True)
