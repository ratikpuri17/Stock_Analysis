import os
import nltk
import spacy
# spacy.load('en')
# from spacy.lang.en import English
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import tweepy
import json
import pandas as pd
import numpy as np
import random
import gensim
from gensim import corpora
from wordcloud import WordCloud

# parser = English()
nlp=spacy.load('en_core_web_sm')


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('URL')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


# nltk.download('wordnet')


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)




def prepare_text_for_lda(text):

    en_stop = set(nltk.corpus.stopwords.words('english'))
    stop_list = ["Mrs.","Ms.","say","WASHINGTON","'s","Mr.","@yahoofinance","URL"]

    for w in stop_list:
        en_stop.add(w)

    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens



consumer_key = os.environ['TWITTER_CONSUMER_KEY']
consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
access_token = os.environ['TWITTER_ACCESS_TOKEN']
access_secret = os.environ['TWITTER_ACCESS_SECRET']


def fetch_trend():

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
    tweets = api.user_timeline(screen_name='YahooFinance', count=100)
    #json.dumps(status)

    t=[]
    for tweet in tweets:

        t.append(tweet['text'])


    tweet_data=[]

    for tweet in t:
        tokens = prepare_text_for_lda(tweet)
        if random.random() > .99:
            print(tokens)
        tweet_data.append(tokens)


    dictionary = corpora.Dictionary(tweet_data)
    corpus = [dictionary.doc2bow(text) for text in tweet_data]

    NUM_TOPICS = 8
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    # ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=10)
    # for topic in topics:
    #     print(topic)

    return all_words(ldamodel)


def all_words(ldamodel):

    l=ldamodel.show_topics(num_topics=10, num_words=20, log=False, formatted=False)
#     print(len(l))
#     print(l)

    all_words=[]

    for topic in range(len(l)):
        
        words_array=l[topic][1]
        
        for words in words_array:
            all_words.append(words[0])
        
        

    s=set(all_words)
    
    g=list(s)


    return Word_Cloud(g)


def Word_Cloud(l):
    long_string = ','.join(l)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white",height=700,width=700, max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()

    return wordcloud
    

