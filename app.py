from flask import Flask, render_template, jsonify, send_file
from pymongo import MongoClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import nltk
from collections import Counter

app = Flask(__name__)

# Download the NLTK data (run this once)
nltk.download('opinion_lexicon')
# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client.SentimentAnalysis  # Update with your database name
collection = db.tweets  # Update with your collection name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sentiments')
def get_sentiments():
    limit = 17000
    sentiments_data = {"Positive": 0, "Negative": 0, "Neutral": 0}

    # Iterate through the first 100 entries
    for tweet in collection.find().limit(limit):
        sentiment = tweet["sentiment"]
        sentiments_data[sentiment] += 1
    sentiments_array = [{"sentiment": key, "count": value} for key, value in sentiments_data.items()]
  
    return jsonify(sentiments_array)

def find_most_repeating_positive_words(limit=5):
    word_count = Counter()

    # Load positive words from NLTK's opinion lexicon
    positive_word_list = set(nltk.corpus.opinion_lexicon.positive())

    # Fetch positive sentences from the MongoDB collection (limit to the first 10 entries)
    positive_sentences = [tweet["tweet"].lower() for tweet in collection.find({"sentiment": "Positive"}).limit(100)]

    # Iterate through the sentences and count positive word occurrences
    for sentence in positive_sentences:
        # Assuming the sentences are lowercase
        words = sentence.split()
        positive_words = [word for word in words if word in positive_word_list]
        word_count.update(positive_words)

    # Get the most common positive words
    most_common_positive_words = word_count.most_common(limit)

    return most_common_positive_words

def generate_wordcloud_image(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words))
    
    # Save the word cloud image using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Save the image to a BytesIO buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    return img_buffer

@app.route('/download_wordcloud')
def download_wordcloud():
    most_common_positive_words = find_most_repeating_positive_words(limit=20)
    
    # Generate word cloud image
    wordcloud_image_buffer = generate_wordcloud_image(most_common_positive_words)
    
    # Send the image file for download
    return send_file(wordcloud_image_buffer, as_attachment=True, download_name='wordcloud.png', mimetype='image/png')

@app.route('/get_positive_words')
def get_positive_words():
    most_common_positive_words = find_most_repeating_positive_words(limit=20)
    return jsonify(most_common_positive_words)

if __name__ == '__main__':
    app.run(debug=True)
