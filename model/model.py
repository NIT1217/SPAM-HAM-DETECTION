import os
import re
import pickle
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. NLTK STOPWORDS
# ============================================================

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


# ============================================================
# 2. PROJECT PATH
# ============================================================

# model.py is inside:
# SPAM-HAM-DETECTION/model/model.py
#
# BASE_DIR becomes:
# SPAM-HAM-DETECTION/

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

dataset_path = os.path.join(
    BASE_DIR,
    "dataset",
    "spamham.csv"
)

print("Dataset path:")
print(dataset_path)

data = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print(data.head())
print("\nDataset shape:", data.shape)


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

corpus = []

for i in range(len(data)):

    review = re.sub(
        "[^a-zA-Z]",
        " ",
        data["Message"].iloc[i]
    )

    review = review.lower()

    review = review.split()

    review = [
        ps.stem(word)
        for word in review
        if word not in stop_words
    ]

    review = " ".join(review)

    corpus.append(review)


print("\nText preprocessing completed!")


# ============================================================
# 5. LABEL ENCODING
# ============================================================

le = LabelEncoder()

y = le.fit_transform(
    data["Label"].values
)

print("\nLabel classes:")
print(le.classes_)

# ham  -> 0
# spam -> 1


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    corpus,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train_text))
print("Test size:", len(X_test_text))


# ============================================================
# 7. TF-IDF FEATURE EXTRACTION
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000
)

# IMPORTANT:
# fit_transform ONLY on training data

X_train = tfidf.fit_transform(
    X_train_text
)

# Only transform test data

X_test = tfidf.transform(
    X_test_text
)

print("\nTF-IDF completed!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ============================================================
# 8. HYPERPARAMETER TUNING
# ============================================================

params = {
    "alpha": [
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
        10.0
    ]
}

mnb_model = MultinomialNB()

mnb_cv = GridSearchCV(
    estimator=mnb_model,
    param_grid=params,
    cv=5,
    scoring="accuracy"
)

print("\nRunning GridSearchCV...")

mnb_cv.fit(
    X_train,
    y_train
)

print("\nBest Parameters:")
print(mnb_cv.best_params_)

print("\nBest Cross Validation Accuracy:")
print(mnb_cv.best_score_)


# ============================================================
# 9. TRAIN FINAL MODEL
# ============================================================

spam_detect_model = MultinomialNB(
    **mnb_cv.best_params_
)

spam_detect_model.fit(
    X_train,
    y_train
)

print("\nFinal model trained successfully!")


# ============================================================
# 10. PREDICTION
# ============================================================

y_pred = spam_detect_model.predict(
    X_test
)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

confusion_m = confusion_matrix(
    y_test,
    y_pred
)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_m)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    )
)


# ============================================================
# 12. MODEL DIRECTORY
# ============================================================

model_dir = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(
    model_dir,
    exist_ok=True
)


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

model_path = os.path.join(
    model_dir,
    "spam_detect_model.pkl"
)

with open(
    model_path,
    "wb"
) as file:

    pickle.dump(
        spam_detect_model,
        file
    )


# ============================================================
# 14. SAVE TF-IDF VECTORIZER
# ============================================================

vectorizer_path = os.path.join(
    model_dir,
    "vectorizer.pkl"
)

with open(
    vectorizer_path,
    "wb"
) as file:

    pickle.dump(
        tfidf,
        file
    )


# ============================================================
# 15. SAVE LABEL ENCODER
# ============================================================

label_encoder_path = os.path.join(
    model_dir,
    "label_encoder.pkl"
)

with open(
    label_encoder_path,
    "wb"
) as file:

    pickle.dump(
        le,
        file
    )


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("FILES SAVED SUCCESSFULLY")
print("========================================")

print("\nModel:")
print(model_path)

print("\nVectorizer:")
print(vectorizer_path)

print("\nLabel Encoder:")
print(label_encoder_path)