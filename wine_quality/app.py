import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('wine_quality.csv')

# Preprocess the dataset
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

df = df.drop(['type','total sulfur dioxide', 'free sulfur dioxide', 'volatile acidity', 'chlorides', 'fixed acidity', 'citric acid','sulphates'], axis=1)

df['best quality'] = [1 if x >= 7 else 0 for x in df['quality']]
df.replace({'white': 1, 'red': 0}, inplace=True)
df = df.drop('quality', axis=1)

# Split the dataset into features and target
target = df['best quality']
features = df.drop('best quality', axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.3, random_state=40)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain, ytrain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    residual_sugar = float(request.form['Residual sugar'])
    density = float(request.form['Density'])
    ph = float(request.form['pH'])
    alcohol = float(request.form['Alcohol'])
    # Add more features as needed

    input_data = np.array([[residual_sugar,density,ph,alcohol]])  # Add more features here

    prediction = knn.predict(input_data)[0]

    quality_message = "Excellent" if prediction == 1 else "Average"

    return render_template('result.html', quality_message=quality_message)

if __name__ == '__main__':
    app.run(debug=True)
