from collections import Counter
import nltk
from pymongo import MongoClient

# Download the NLTK data (run this once)
nltk.download('opinion_lexicon')

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client.SentimentAnalysis  # Replace with your actual database name
collection = db.tweets  # Replace with your actual collection name

def find_most_repeating_positive_words(limit=20):
    word_count = Counter()

    # Load positive words from NLTK's opinion lexicon
    positive_word_list = set(nltk.corpus.opinion_lexicon.negative())

    # Fetch positive sentences from the MongoDB collection (limit to the first 10 entries)
    positive_sentences = [tweet["tweet"].lower() for tweet in collection.find({"sentiment": "Positive"}).limit(17000)]

    # Iterate through the sentences and count positive word occurrences
    for sentence in positive_sentences:
        # Assuming the sentences are lowercase
        words = sentence.split()
        positive_words = [word for word in words if word in positive_word_list]
        word_count.update(positive_words)

    # Get the most common positive words
    most_common_positive_words = word_count.most_common(limit)

    return most_common_positive_words

most_common_positive_words = find_most_repeating_positive_words(limit=20)
print("Most Common Positive Words:")

print(most_common_positive_words)
for word, count in most_common_positive_words:
    print(f"{word}: {count}")
