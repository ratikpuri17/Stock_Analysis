
# coding: utf-8

# In[1]:


import spacy
spacy.load('en')
from spacy.lang.en import English

parser = English()


# In[2]:



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


# In[3]:


import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


# In[4]:


en_stop = set(nltk.corpus.stopwords.words('english'))


# In[5]:


stop_list = ["Mrs.","Ms.","say","WASHINGTON","'s","Mr.","@yahoofinance","URL"]


# In[6]:


for w in stop_list:
    en_stop.add(w)


# In[7]:


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


# In[8]:


import tweepy
import json
import pandas as pd
import numpy as np
import pandas as pd
import os

# In[9]:
consumer_key = os.environ['TWITTER_CONSUMER_KEY']
consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
access_token = os.environ['TWITTER_ACCESS_TOKEN']
access_secret = os.environ['TWITTER_ACCESS_SECRET']



# In[10]:


# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)


# In[11]:


api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
tweets = api.user_timeline(screen_name='YahooFinance', count=100)
#json.dumps(status)


# In[12]:


t=[]
for tweet in tweets:

    t.append(tweet['text'])


# In[13]:


t[1]


# In[14]:


import random

tweet_data=[]

for tweet in t:
    tokens = prepare_text_for_lda(tweet)
    if random.random() > .99:
        print(tokens)
    tweet_data.append(tokens)


# In[15]:


tweet_data[0]


# In[16]:


from gensim import corpora
dictionary = corpora.Dictionary(tweet_data)
corpus = [dictionary.doc2bow(text) for text in tweet_data]


# In[17]:


import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


# In[18]:


import gensim
NUM_TOPICS = 8
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)


# In[19]:


l=ldamodel.show_topics(num_topics=10, num_words=10, log=False, formatted=False)


# In[20]:


l[1]


# In[21]:


l[2][1][0][0]


# In[22]:


for i in range(len(l)):
    print(l[i][1][2])


# In[23]:


words=[]

for topic in range(len(l)):
    
    topics=[]
    tp=l[topic][1]
    
    for j in range(len(l)):
        
        topics.append(tp[j][0])
        
    
    
    words.append(topics)
    
    
    


# In[24]:


words[4]


# In[25]:


s=set()
for i in range(len(words)):
    
    for j in words[i]:
        
        s.add(j)


# In[26]:


s


# In[27]:


l=list(s)


# In[29]:


# !pip install WordCloud


# In[34]:


# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(l)
# Create a WordCloud object
wordcloud = WordCloud(background_color="white",height=700,width=700, max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# In[167]:


wordcloud.to_file('cloud.png')

