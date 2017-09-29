import plotoutput
from nltk.corpus import sentiwordnet as swn  # importing the sentiwordnet
import json  # load json files
import nltk  # natural language processing toolkit
import frequentItemset
from nltk.corpus import stopwords  # Corpus for stopwords

nltk.data.path.append("C:/Users/Purva/Anaconda3/Lib/site-packages/nltk/corpus/reader")
nltk.data.path.append("C:/Users/Purva/Anaconda3/Lib/site-packages/nltk/corpora/sentiwordnet")
nltk.data.path.append("C:/Users/Purva/Anaconda3/Lib/site-packages/nltk/corpora/sentiwordnet")
nltk.data.path.append("C:/Users/Purva/Anaconda3/Lib/site-packages/nltk")
_author_ = 'purva'
_project_ = 'Sentiment Analysis'

# Valid ASIN for input
# B00004THCZ
# 1400501520
# 0528881469
# B00004U8K4
# 0972683275

with open('Electronics.json') as data_file1:
    data = json.load(data_file1)
data_file1.close()

stop = set(stopwords.words('english'))
listreviews1 = []

asin = input("Enter the asin of product:")
print("Analyzing", end="")
for review in data:
    if review['asin'] == asin:
        listreviews1.append(review['reviewText'])
tags = []
count = 0
listreviews = []
for sentence in listreviews1:
    text = nltk.word_tokenize(sentence)
    buffer = []
    for w in text:
        if w not in stop:
            buffer.append(w)
    tags.append(nltk.pos_tag(buffer))
    listreviews.append(buffer)

total = len(listreviews)
count = count + len(text)

print(".", end="")

newtags = []
for tag in tags:
    for t in tag:
        if t[1] == 'NN' or t[1] == 'NNS' or t[1] == 'JJ' or t[1] == 'JJR' or t[1] == 'JJS':
            newtags.append(t)

print(".", end="")
candidatefeatures = []
for tag in newtags:
    if tag[1] == 'NN' or tag[1] == 'NNS':
        candidatefeatures.append(tag)

freqItems = frequentItemset.findFrequent(candidatefeatures, total)

print(".", end="")

freqItems = frequentItemset.infrequentFeatures(listreviews1, freqItems)

print(".", end="")

opinionwords = frequentItemset.extractOpinion(freqItems, newtags)

print(".", end="")

sentiment = []
for word, tag, adj in opinionwords:
    if isinstance(adj, tuple):
        # print(".", end="")
        synset = list(swn.senti_synsets(adj[0], "a"))
        if len(synset) > 0:
            t = word, synset[0]
            sentiment.append(t)
    else:
        for t in adj:
            # print(".", end="")
            synset = list(swn.senti_synsets(t[0], "a"))
            if len(synset) > 0:
                t = word, synset[0]
                sentiment.append(t)
print(".")
sentimentList = []
for feature, postag in freqItems:
    pos = 0
    neg = 0
    count = 0
    for word, sent in sentiment:
        if word == feature:
            pos = pos + sent.pos_score()
            neg = neg + sent.neg_score()
            count = count + 1
    if pos > 0:
        pos = float(pos / count)
    if neg > 0:
        neg = float(neg / count)
    if neg > 0 or pos > 0:
        print(feature + ":Positive = " + str(pos) + ", Negative = " + str(neg))
        t = feature, pos, neg
        sentimentList.append(t)
plotoutput.plotSentiment(sentimentList, asin)
