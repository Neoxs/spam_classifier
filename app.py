from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

# load the model
clf = pickle.load(open("./utils/spam_classifier.sav", 'rb'))
# Load dataset
df = pd.read_csv("./utils/spam.csv", encoding='latin-1')
# Fit the training data and then return the matrix
count_vector = CountVectorizer()
count_vector.fit_transform(df['v2'])

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    email = data['email']
    # vectorize the email words
    input_vector = count_vector.transform([email])
    # calculate prediction
    prediction = clf.predict(input_vector)
    print(prediction)
    return 'spam' if prediction[0] == 1 else 'ham'