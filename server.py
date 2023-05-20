import re
import urllib
import json
import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
import torch.nn as nn
import texthero as hero
from urllib.parse import urlsplit
import requests
import torch
from bs4 import BeautifulSoup
from requests import get
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from huggingface_hub import from_pretrained_keras
import spacy_sentence_bert
import csv 
import asyncio
import aiohttp

from flask import Flask,render_template, request, jsonify
from flask_cors import CORS

nlp = spacy_sentence_bert.load_model('en_nli_roberta_base')
model = from_pretrained_keras("keras-io/bert-semantic-similarity")
labels = ["contradiction", "entailment", "neutral"]


import warnings
warnings.filterwarnings('ignore')

from timeit import default_timer as timer

import math
from GoogleNews import GoogleNews
from datetime import datetime

googlenews = GoogleNews(lang='en')
googlenews.set_encode('utf-8')

def get_links_from_google_news(claim, top=10):
    googlenews.search(claim)
    
    results = googlenews.results()
    googlenews.clear()
    
    links = [x['link'] for x in results]
    
    return links

def get_sentences_from_link(link, text, top=10):
    request = requests.get(link, verify=False, timeout=20)
    Soup = BeautifulSoup(request.text, 'lxml')
    
    heading_tags = ['p']


    results = []
    used = []

    for tags in Soup.find_all(heading_tags):
        if 'h' in tags.name:
            tokens = tags.text.strip().split()
            if len(tokens) > 8:
                if tags.text.strip() not in used:
                    used.append(tags.text.strip())
                    results.append([tags.name, tags.text.strip()])
        else:
            tokens = tags.text.strip().split()
            if len(tokens) > 8:
                if tags.text.strip() not in used:
                    used.append(tags.text.strip())
                    results.append([tags.name, tags.text.strip()])
    doc1 = nlp(text)
    sim = []
    for r in results:
        sim.append(doc1.similarity(nlp(r[1])))
    zipped = zip(sim, results)
    zipped = sorted(zipped, reverse=True)
    high_conf = [a for s, a in zipped if s >= 0.6]

    return high_conf[:top], request.url

def scrap_evidences(text, links):
    new_links = []
    for link in links:
        conf, lin = get_sentences_from_link(link, text)
        new_links.append([lin, conf])
    return new_links

def concatenate_evidences(claim, links):
    summ = []
    for link in links:
        if type(link[1]) == list:
            for text in link[1]:
                if type(link[1]) == list:
                    summ.append(text[1])
                else:
                    summ.append(text)
        elif type(link[1]) == str:
            summ.append(link[1])

    urls = re.findall(r'https?:\/+\/+t+\.+co+\/+\S*', claim)
    
    for li in urls:
        claim = claim.replace(li, '')
    claim = claim.strip()

    if summ:
        summary = (claim, ' '.join(summ).replace('\n', '').replace('\t', ''))
    else:
        summary = ('', '')

    return summary

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data."""
    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=32,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

def predict(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    probs = model.predict(test_data[0])[0]
    
    labels_probs = {labels[i]: float(probs[i]) for i, _ in enumerate(labels)}
    return labels_probs     

def fake_news_detection(claim):

    links = get_links_from_google_news(claim)
    
    evidence_list = scrap_evidences(claim, links[:10])
   
    evidence = concatenate_evidences(claim, evidence_list)
    
    k = predict(claim, evidence[1])
    print(k)

    return (k)

app = Flask(__name__)
app.config['DEBUG'] = False
CORS(app)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

async def prediction(title):
    # Perform asynchronous calls to your ML model for prediction
    # Replace the placeholders with actual code to process the title and get the prediction scores
    prediction_scores = fake_news_detection(title)

    return prediction_scores

#async def handle_request(titles):
#    tasks = [prediction(title) for title in titles]
##    prediction_scores = await asyncio.gather(*tasks)
    return prediction_scores


@app.route('/predict', methods=['POST'])
def handle_predict():
    title = request.form.get('title') # Get the title from the request form
    
    # Call the predict function asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    prediction_scores = loop.run_until_complete(prediction(title))
    loop.close()

    # Return the prediction scores as a JSON response
    return jsonify(prediction_scores)
    

if __name__ == '__main__':
    app.run(port=7000, host='0.0.0.0')
    
    