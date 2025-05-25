import pickle
from flask import Flask, render_template, request

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment_svc_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)
    
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    review = ""
    if( request.method == 'POST'):
        review = request.form['review']
        vec = vectorizer.transform([review])
        if selector:
            vec = selector.transform(vec)
        pred = model.predict(vec)[0]
        prediction = "Positive" if pred == 1 else "Negative"
    
    return render_template('index.html', prediction=prediction, review=review)

if __name__ == '__main__':
    app.run(debug=True)