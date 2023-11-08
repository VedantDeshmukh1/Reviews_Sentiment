#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')


# In[6]:


import pandas as pd
from pymongo import MongoClient


# In[7]:


df = pd.read_csv("test.csv")
sample = df["tweet"][0]
sample
df


# In[8]:


client = MongoClient("mongodb://localhost:27017/SentimentAnalysis")
db = client["SentimentAnalysis"]
collection = db["tweets"]


# In[ ]:


from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from scipy.special import softmax
# Loading the pre-trained RoBERTa model and tokenizer
model_name = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

#searching the database iteratively
for tweet in collection.find():
    tweet_text = tweet["tweet"]
    tweet_id = tweet["_id"]
    encoded_text = tokenizer(tweet_text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
    'Negative' : scores[0],
    'Neutral' : scores[1],
    'Positive' : scores[2]
        }
    max_value = None
    max_keys = []

    # Loop through the dictionary to find the maximum value and key(s)
    for key, value in scores_dict.items():
        if max_value is None or value > max_value:
            max_value = value
            max_keys = [key]
        elif value == max_value:
            max_keys.append(key)
    sentiment_label = max_keys
    #In the exisiting dataset now we will append the sentiments of the comments being pos,neg or neutral
    collection.update_one(
        {"_id": tweet_id},
        {"$set": {"sentiment": sentiment_label[0]}}
    )
    


# In[ ]:


pos_count = 0
neu_count = 0
neg_count = 0

for tweet in collection.find():
    senti = tweet["sentiment"]
    if(senti == "Positive"):
        pos_count +=1
    if(senti == "Neutral"):
        neu_count +=1
    if(senti == "Negative"):
        neg_count +=1


# In[ ]:


print(pos_count)


# In[ ]:




