from flask import Flask, render_template, request
import pickle

from pymongo import MongoClient
from dotenv import load_dotenv

import os
from datetime import datetime

from recommendation import get_recommendations


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")


# ==========================================
# CHECK MONGO URI
# ==========================================

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the .env file")


# ==========================================
# CONNECT TO MONGODB
# ==========================================

client = MongoClient(MONGO_URI)

db = client["spam_ham_db"]

search_history = db["search_history"]


# ==========================================
# TEST MONGODB CONNECTION
# ==========================================

try:

    client.admin.command("ping")

    print("MongoDB connected successfully!")

except Exception as e:

    print("MongoDB connection failed:", e)


# ==========================================
# LOAD SPAM/HAM MODEL
# ==========================================

with open("model/spam_detect_model.pkl", "rb") as file:

    spam_detect_model = pickle.load(file)


# ==========================================
# LOAD VECTORIZER
# ==========================================

with open("model/vectorizer.pkl", "rb") as file:

    vectorizer = pickle.load(file)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("home.html")


# ==========================================
# INPUT PAGE
# ==========================================

@app.route("/input")
def input_page():

    return render_template("input.html")


# ==========================================
# PREDICTION + RECOMMENDATION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------
    # Get input text
    # --------------------------------------

    input_text = request.form.get("input_text")


    # --------------------------------------
    # Check empty input
    # --------------------------------------

    if not input_text or not input_text.strip():

        return render_template("input.html")


    # Remove unnecessary spaces
    input_text = input_text.strip()


    # ======================================
    # SPAM / HAM PREDICTION
    # ======================================

    input_vector = vectorizer.transform([input_text])

    prediction = spam_detect_model.predict(input_vector)[0]


    # --------------------------------------
    # Convert prediction to readable result
    # --------------------------------------

    if prediction == 1:

        result = "spam"

    else:

        result = "ham"


    # ======================================
    # GET PREVIOUS SEARCH HISTORY
    # ======================================

    history = list(

        search_history.find(

            {},

            {
                "_id": 0,
                "input_text": 1,
                "prediction": 1
            }

        )

    )


    # ======================================
    # GENERATE RECOMMENDATIONS
    # ======================================

    recommendations = get_recommendations(

        input_text,

        history,

        top_n=3

    )


    # ======================================
    # SAVE CURRENT SEARCH
    # ======================================

    search_history.insert_one({

        "input_text": input_text,

        "prediction": result,

        "timestamp": datetime.now()

    })


    print("Search history saved to MongoDB.")


    # ======================================
    # SEND RESULT + RECOMMENDATIONS
    # TO prediction.html
    # ======================================

    return render_template(

        "prediction.html",

        input_text=input_text,

        prediction=result,

        recommendations=recommendations

    )


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)