from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string
import pandas as pd  # Import pandas to read the CSV file

app = Flask(__name__)

# Allow all origins for development (change this for production!)
CORS(app, origins="*")  # This allows any origin to make requests to your backend

# Load the sentiment model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load the Yelp data
try:
    yelp_data = pd.read_csv('yelp.csv')
    print("Yelp data loaded successfully.")
except Exception as e:
    print(f"Error loading Yelp data: {e}")

# Preprocessing function for text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Endpoint to fetch all reviews
@app.route('/reviews', methods=['GET'])
def get_reviews():
    # Check if the yelp_data contains the necessary columns
    if 'text' not in yelp_data.columns or 'stars' not in yelp_data.columns:
        return jsonify({'error': 'Invalid data format, missing review or rating columns'}), 400

    # Extract review text and ratings
    reviews = yelp_data[['text', 'stars']].to_dict(orient='records')
    
    # Clean the 'stars' field by removing any non-numeric characters (if needed)
    for review in reviews:
        review['stars'] = re.sub(r'\D', '', review['stars'])  # Remove non-numeric characters (in case there's anything extra)
        review['stars'] = int(review['stars'])  # Ensure it's an integer

    # If no reviews, return an error message
    if not reviews:
        return jsonify({'message': 'No reviews available'}), 404
    
    return jsonify(reviews)

# Endpoint for sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    review = request.json.get('review')
    
    if not review:
        return jsonify({'error': 'Please enter a review'}), 400

    # Preprocess the review text
    cleaned_review = clean_text(review)
    review_tfidf = vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = model.predict(review_tfidf)  # Model prediction (e.g., 0 or 1)

    # Convert prediction to sentiment label
    sentiment_label = 'positive' if prediction[0] == 1 else 'negative'

    # Create a custom response based on sentiment
    if sentiment_label == 'positive':
        response_message = "Thank you for your kind words! We're thrilled to know you had a great experience."
    else:
        response_message = "Thank you for your feedback! We're sorry to hear this and will work on improving."

    # Return both sentiment and the custom message
    return jsonify({'sentiment': sentiment_label, 'message': response_message})

if __name__ == '__main__':
    app.run(debug=True)







