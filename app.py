from flask import Flask, render_template, request
import pickle

from pymongo import MongoClient
from dotenv import load_dotenv

import os
from datetime import datetime

from reccomendation.reccomendation import get_recommendations


# ==========================================
# FLASK APPLICATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv(os.path.join(BASE_DIR, ".env"))

MONGO_URI = os.getenv("MONGO_URI")


# ==========================================
# CHECK MONGO URI
# ==========================================

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the .env file")


# ==========================================
# CONNECT TO MONGODB
# ==========================================

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

db = client["spam_ham_db"]

search_history = db["search_history"]
recommendation_collection = db["recommendations"]


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

model_path = os.path.join(
    BASE_DIR,
    "model",
    "spam_detect_model.pkl"
)

with open(model_path, "rb") as file:

    spam_detect_model = pickle.load(file)


# ==========================================
# LOAD VECTORIZER
# ==========================================

vectorizer_path = os.path.join(
    BASE_DIR,
    "model",
    "vectorizer.pkl"
)

with open(vectorizer_path, "rb") as file:

    vectorizer = pickle.load(file)


print("Spam/Ham model loaded successfully!")
print("Vectorizer loaded successfully!")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def index():

    return render_template("index.html")


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


    input_text = input_text.strip()


    # ======================================
    # SPAM / HAM PREDICTION
    # ======================================

    input_vector = vectorizer.transform([input_text])

    prediction = spam_detect_model.predict(input_vector)[0]


    # --------------------------------------
    # Convert prediction
    # --------------------------------------

    if prediction == 1:

        result = "spam"

    else:

        result = "ham"


    print("Prediction:", result)


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


    print("Recommendations generated:", recommendations)


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
    # SAVE RECOMMENDATIONS TO MONGODB
    # ======================================

    for recommendation in recommendations:

        recommendation_collection.insert_one({

            "input_text": input_text,

            "recommendation_text": recommendation["text"],

            "similarity_score": float(
                recommendation["score"]
            ),

            "prediction": result,

            "timestamp": datetime.now()

        })


    print("Recommendations saved to MongoDB.")


    # ======================================
    # SEND RESULT TO prediction.html
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

    app.run(host ="0.0.0.0",debug=True)