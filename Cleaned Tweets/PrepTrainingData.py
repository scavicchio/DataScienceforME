#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://programminghistorian.org/en/lessons/sentiment-analysis
# https://textblob.readthedocs.io/en/dev/quickstart.html#spelling-correction
# https://www.kaggle.com/ssishu/factual-authenticity-analysis-of-tweets
# https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f



# In[2]:


# import the training tweets 
import pandas as pd
df = pd.read_csv("resources/trainingSet.csv",encoding='latin-1', header=None)
df.columns = ["sentiment", "id", "date", "flag", "user", "text"]

# making sure the tweets are text
df['text'] = df['text'].astype(str)

# cleaning columns I don't think we will need
df = df.drop(columns=['id','date','flag','user']);

# adding a new column to save the processed text
df['cleanText'] = df['text'].astype(str)

print(df.head())

import string
def removePunctuation(text):
    # need to do this as would rather have two words than a single
    # word that is supposed to be hyphenated
    # this is kinda slow
    #text = text.replace('-',' ')
    #return "".join([x for x in text if x not in string.punctuation])
    # new method below is faster ?
    return text.translate(str.maketrans('', '', string.punctuation))

import re
def removeMentionsAndURLs(text):
    mentions = r'@[A-Za-z0-9_]+'
    url1 = r'https?://[^ ]+'
    url2 = r'www.[^ ]+' 
    comb = r'|'.join((mentions, url1))
    comb = r'|'.join((comb, url2))
    return re.sub(comb,'',text)

def makeLowercase(text):
    return text.lower()

import nltk
# nltk.download('stopwords') # you only need this if you dont have them downloaded already
# to get everything in nltk, do 
# nltk.download('all')
from nltk.corpus import stopwords
def removeStopwords(text): 
    return  ' '.join([x for x in text.split() if x not in stopwords.words('english')])

# 3 different options
from textblob import TextBlob
from autocorrect import Speller 
from spellchecker import SpellChecker

spell = Speller(lang='en')
def correctSpelling(text): 
    #print("ORIGINAL")
    #print(text)
    #theReturn = spell(text)
    theReturn = str(TextBlob(text).correct())
    return theReturn # about 70% accurate

spell2 = SpellChecker()
def correctSpelling2(text):
    corrected_text = []
    misspelled_words = spell2.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell2.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

# this doesn't really work yet
from pycontractions import Contractions
cont = Contractions('resources/GoogleNews-vectors-negative300.bin')

def expandContractions(text): 
    global cont
    theReturn = list(cont.expand_texts([text], precise=True))[0]
    return theReturn

#nltk.download('youll need to put the lemmatizer here')
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer() 

def lemmatizeString(text):
    theReturn = ''
    words = word_tokenize(text)
    for w in words:
        theReturn += lemmatizer.lemmatize(w) + ' '

    return theReturn

# In[4]:
count = 0
from timeit import default_timer as timer

# apply the cleaning functions in a speific order
# this takes a long fucking time
def cleanTweetText(text):
    # doing the removals first in an attempt to speed up computation
    #print("removeMentionsAndURLs")
    #start = timer()
    text = expandContractions(text)

    text = removeMentionsAndURLs(text)

    #end = timer()
    #print(end - start) 
    #print("removeStopwords")
    #start = timer()

    text = removeStopwords(text) #works

    #end = timer()
    #print(end - start) 
    #print("removePunctuation")
    #start = timer()

    text = removePunctuation(text) #works

    #end = timer()
    #print(end - start) 
    #print("correctSpelling")
    #start = timer()

    #NOT CORRECTING SPELLING RIGHT NOW CAUSE IT TAKES FOREVER
    #text = correctSpelling(text) #works

    #end = timer()
    #print(end - start) 
    #print("makeLowercase")
    #start = timer()

    text = makeLowercase(text) #works

    #end = timer()
    #print(end - start) 
    #print("lemmatizeString")
    #start = timer()

    text = lemmatizeString(text)

    #end = timer()
    #print(end - start) 

    #text = expandContractions(text) # doesnt work yet - womp womp
    global count
    count += 1
    if (count % 1000 == 0):
        print(count)
        print(text)
    return text



count = 0
print("cleaning training set data")
df['cleanText'] = df['cleanText'].apply(lambda x : cleanTweetText(x))
df.to_csv('cleanTrainingSet'.csv)


print("correcting Spelling")
df_gen['spellCorrected'] = df_gen['cleanText'].astype(str)
df['spellCorrected'] = df['spellCorrected'].apply(lambda x : correctSpelling2(x))

df.to_csv('cleanTrainingSet'.csv)


