#!/usr/bin/env python
# coding: utf-8

# In[1]:


import GetOldTweets3 as got
import pandas as pd
from itertools import compress 
from datetime import timedelta, date


# ## Get tweets data using API text_query

# In[2]:


# input query topics, number of counts to see, date of tweets(every other day from 2020-1-1 to 2020-4-25)
def text_query_to_csv(text_query, count, since_until):
    
    # Creation of query object (get )
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setSince(since_until[0]).setUntil(since_until[1]).setMaxTweets(count).setTopTweets(True)
   
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    
    # Creating list of chosen tweet data (tweets date, tweets text, user sent the tweets, retweets number)
    text_tweets = [[tweet.date, tweet.text, tweet.username, tweet.retweets] for tweet in tweets]
    
    # Creation of dataframe from tweets
    tweets_df = pd.DataFrame(text_tweets, columns = ['Datetime', 'Text','user','retweets'])
    
    return tweets_df
    
    # Converting tweets dataframe to csv file


# In[3]:


# input query topics, number of counts to see, date of tweets(every other day from 2020-1-1 to 2020-4-25)
def popular_to_csv(count, since_until):
    # Creation of query object (get )
    query = ' '
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(since_until[0]).setUntil(since_until[1]).setMaxTweets(count).setTopTweets(True)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    # Creating list of chosen tweet data (tweets date, tweets text, user sent the tweets, retweets number)
    text_tweets = [[tweet.date, tweet.text, tweet.username, tweet.retweets] for tweet in tweets]
    # Creation of dataframe from tweets
    tweets_df = pd.DataFrame(text_tweets, columns = ['Datetime', 'Text','user','retweets'])
    return tweets_df
    
    # Converting tweets dataframe to csv file


# In[4]:


# input query topics, number of counts to see, date of tweets(every other day from 2020-1-1 to 2020-4-25)
def text_location_query_to_csv(text_query, location, within, count, since_until):
    
    # Creation of query object (get )
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setNear(location).setWithin(within).setSince(since_until[0]).setUntil(since_until[1]).setMaxTweets(count).setTopTweets(True)
   
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    
    # Creating list of chosen tweet data (tweets date, tweets text, user sent the tweets, retweets number)
    text_tweets = [[tweet.date, tweet.text, tweet.username, tweet.retweets] for tweet in tweets]
    
    # Creation of dataframe from tweets
    tweets_df = pd.DataFrame(text_tweets, columns = ['Datetime', 'Text','user','retweets'])
    
    return tweets_df
    
    # Converting tweets dataframe to csv file


# In[5]:


# input query topics, number of counts to see, date of tweets(every other day from 2020-1-1 to 2020-4-25)
def popular_location_query_to_csv(location, within, count, since_until):
    
    # Creation of query object (get )
    tweetCriteria = got.manager.TweetCriteria().setNear(location).setWithin(within).setSince(since_until[0]).setUntil(since_until[1]).setMaxTweets(count).setTopTweets(True)
   
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    
    # Creating list of chosen tweet data (tweets date, tweets text, user sent the tweets, retweets number)
    text_tweets = [[tweet.date, tweet.text, tweet.username, tweet.retweets] for tweet in tweets]
    
    # Creation of dataframe from tweets
    tweets_df = pd.DataFrame(text_tweets, columns = ['Datetime', 'Text','user','retweets'])
    
    return tweets_df
    
    # Converting tweets dataframe to csv file


# In[6]:


# creating a date loop for the required date range
date_loop = []  
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = date(2020, 1, 21)
end_date = date(2020, 5, 5)
for single_date in daterange(start_date, end_date):
    date_loop.append(single_date.strftime("%Y-%m-%d"))  


# In[7]:


# creating pairs of date from the date range
res = list(zip(date_loop, date_loop[1:] + date_loop[:1])) 


# In[8]:


# get all popular coronavirus tweets
dic1 = []
nToCollect = 100

for num in range(0,len(res)):
    print(num)
    dic1.append(text_query_to_csv('coronavirus', nToCollect, res[num]))
    
df1 = pd.concat(dic1)
df1.to_csv('general_covid_tweet_data.csv')


# In[ ]:


# get all popular general tweets in NYC area
dic3 = []
nToCollect = 100

location = "New York City, New York"
within = "50km"
for num in range(0,len(res)):
    print(num)
    dic3.append(popular_location_query_to_csv(location, within, nToCollect, res[num]))
    
df3 = pd.concat(dic3)
df3.to_csv('nyc_popular_tweet_data.csv')


# In[ ]:


# creating a new dataset for each day's COVID NYC tweets from 1-1 to 4-25
# 50 top tweets for each day for about 110 days
dic4 = []
nToCollect = 100

location = "New York City, New York"
within = "50km"
for num in range(0,len(res)):
    print(num)
    dic4.append(text_location_query_to_csv('coronavirus', location, within, nToCollect, res[num]))
    
df4 = pd.concat(dic4)
df4.to_csv('nyc_covid_tweet_data.csv')

