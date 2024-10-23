from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json


application = app = Flask(__name__)
application.config.from_object(__name__)

loaded_model = None
with open('basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

vectorizer = None
with open('count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"


@application.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    prediction = loaded_model.predict(vectorizer.transform([text]))[0]
    prediction_value = 1 if prediction == "FAKE" else 0

    return jsonify({"prediction": prediction_value})

if __name__ == "__main__":
    application.run(port=5000, debug=True)