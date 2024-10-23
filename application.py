from flask import Flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = app = Flask(__name__)
# load the config
application.config.from_object(__name__)

@application.route('/')
def load_model():
    loaded_model = None
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)
    
    vectorizer = None
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    return loaded_model.predict(vectorizer.transform(['This is fake news']))[0]

if __name__ == '__main__':
    application.run()