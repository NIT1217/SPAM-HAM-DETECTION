from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_recommendations(input_text, history, top_n=3):

    # If there is no history
    if not history:

        return []


    # Extract previous messages
    previous_messages = [

        item["input_text"]

        for item in history

        if "input_text" in item

    ]


    # If no valid previous messages
    if not previous_messages:

        return []


    # Add current message at beginning
    all_messages = [

        input_text

    ] + previous_messages


    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        all_messages
    )


    # Compare current message with previous messages
    similarity_scores = cosine_similarity(

        tfidf_matrix[0:1],

        tfidf_matrix[1:]

    )[0]


    # Store message + similarity
    recommendations = []


    for i, score in enumerate(similarity_scores):

        recommendations.append({

            "text": previous_messages[i],

            "score": float(score)

        })


    # Sort by similarity
    recommendations.sort(

        key=lambda x: x["score"],

        reverse=True

    )


    # Return top recommendations
    return recommendations[:top_n]