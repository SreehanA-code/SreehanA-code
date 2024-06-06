# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values (example: fill with median)
    df.fillna(df.median(), inplace=True)
    
    # Encode categorical variables
    categorical_features = ['ProjectType', 'Team']
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Outcome')  # Remove target variable
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    return df, preprocessor

# Function to train the models
def train_models(df, preprocessor):
    # Split the data into training and testing sets
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline for preprocessing and classification
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(random_state=42))])
    
    # Train the classification model
    clf_pipeline.fit(X_train, y_train)
    
    # Evaluate the classification model
    y_pred = clf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy}")
    
    return clf_pipeline

# Function to predict outcomes
def predict_outcome(model, new_data):
    predictions = model.predict(new_data)
    return predictions

# Function for resource allocation suggestion (dummy implementation)
def suggest_resource_allocation(project_details):
    # Dummy suggestion: evenly distribute resources
    resources = project_details['TeamSize']
    tasks = project_details['Tasks']
    allocation = {f"TeamMember{i+1}": tasks//resources for i in range(resources)}
    return allocation

# Function to identify potential risks (dummy implementation)
def identify_potential_risks(project_details):
    # Dummy risk identification: check for high complexity and low budget
    if project_details['Complexity'] > 7 and project_details['Budget'] < 50000:
        return ["High complexity with low budget"]
    return []
