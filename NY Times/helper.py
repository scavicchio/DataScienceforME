import string
import re
import nltk
from nltk.corpus import stopwords
from pycontractions import Contractions
from textblob import TextBlob
from autocorrect import Speller 
from spellchecker import SpellChecker
from pycontractions import Contractions
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize # import word_tokenize
import joblib

classifier = joblib.load('resources/classifier.pkl')
tfidfVectorizer = joblib.load('resources/tfidfVectorizer.pkl')
cont = Contractions('resources/GoogleNews-vectors-negative300.bin')

def removePunctuation(text):
    # need to do this as would rather have two words than a single
    # word that is supposed to be hyphenated
    # this is kinda slow
    #text = text.replace('-',' ')
    #return "".join([x for x in text if x not in string.punctuation])
    # new method below is faster ?
    return text.translate(str.maketrans('', '', string.punctuation))

def removeMentionsAndURLs(text):
    mentions = r'@[A-Za-z0-9_]+'
    url1 = r'https?://[^ ]+'
    url2 = r'www.[^ ]+' 
    comb = r'|'.join((mentions, url1))
    comb = r'|'.join((comb, url2))
    return re.sub(comb,'',text)

def makeLowercase(text):
    return text.lower()

def removeStopwords(text): 
    return  ' '.join([x for x in text.split() if x not in stopwords.words('english')])


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

def prepText(text):
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

def normalizeSentiment(score):
    if (score == "[4]"):
        return 1
    else: 
        return -1
# feed this function an individual text string and it will output either a 0 or 4
def sentiScore(text):
    corpus = []
    corpus.append(text)
    x_tfid = tfidfVectorizer.transform(corpus).toarray()
    score = normalizeScore(classifier.predict(x_tfid))
    return score

def scoreRawText(text):
	return sentiScore(prepText(text))

 

