from flask import Flask, request
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from flask_cors import CORS
import pickle

# Load Model
model = load_model('LSTMmodel.h5')

# Load Tokenizer
tokenizer = pickle.load(open("tokenizer.h5", 'rb'))


def rate(p):
    return (p*5)


app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return "Review app working"


@app.route("/postreview")
def postreview():
    # Get the data posted from the form
    message = request.get_json(force=True)
    review = message['review']

    # assign the review text to a variable
    a = [review]

    # predict the outcome
    pred = model.predict(pad_sequences(
        tokenizer.texts_to_sequences(a), maxlen=100))
    value = rate(pred.item(0, 0))

    return str(value)[:3]


print("App Running!")

if __name__ == "__main__":
    app.run()