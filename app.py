import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# 🔹 Login Page (First page)
@app.route('/')
def login_page():
    return render_template('login.html')

# 🔹 Login Logic
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == "admin" and password == "1234":
        return render_template('index.html')
    else:
        return render_template('login.html', message="Invalid Login ❌")

# 🔹 Main Project Page (with graph)
@app.route('/home')
def home():
    try:
        df = pd.read_csv("fake_news.csv")

        plt.figure(figsize=(5,4))
        df['label'].value_counts().plot(kind='bar')

        plt.title("Fake vs Real News")
        plt.xlabel("Label (0 = Fake, 1 = Real)")
        plt.ylabel("Count")

        plt.savefig("static/graph.png")
        plt.close()

    except Exception as e:
        print("Error:", e)

    return render_template('index.html')

# 🔹 Prediction
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    cleaned = clean_text(news)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)

    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template('index.html',
                           prediction_text=result,
                           entered_text=news)

# Run app
if __name__ == "__main__":
    app.run(debug=True)