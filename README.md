
# Introduction
In this project, I trained a Logistic Regression Model on Twitter Dataset.
Then I used the model against **real-time** tweets fetched from Twitter

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/06/79592twitter.jpg)


# Dataset
The dataset being used is the "sentiment140" dataset. 
It contains 1,600,000 tweets extracted using the Twitter API. 
The tweets have been annotated (0 = Negative, 4 = Positive) and 
they can be used to detect sentiment.

*The training data isn't perfectly categorised as it has been created 
by tagging the text according to the emoji present. 
So, any model built using this dataset may have lower than expected accuracy, 
since the dataset isn't perfectly categorised.*


The dataset contains the following 6 fields:

- **sentiment:** the polarity of the tweet (0 = negative, 4 = positive)
- **ids:** The id of the tweet
- **date:** the date of the tweet
- **flag:** The query. If there is no query, then this value is NO_QUERY.
- **user:** the user that tweeted
- **text:** the text of the tweet

We will need only the sentiment and text fields, others will be removed.
Moreover, we're changing the sentiment field so that it has new values to reflect the sentiment. (0 = Negative, 1 = Positive)

**Note:** The dataset is obtained from Kaggle, and the link is provided in this repo.

# Install Dependencies
    # utilities
    import re
    import pickle
    import numpy as np
    import pandas as pd

    # plotting
    import seaborn as sns
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # nltk
    from nltk.stem import WordNetLemmatizer

    # sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import confusion_matrix, classification_report

    # twitter api
    import tweepy
    from tweepy import OAuthHandler
# Text Preprocessing
The Preprocessing steps taken are:

- **Lower Casing:** Each text is converted to lowercase.
- **Replacing URLs:** Links starting with "http" or "https" or "www" are replaced by "URL".
- **Replacing Emojis:** Replace emojis by using a pre-defined dictionary containing emojis along with their meaning. (eg: ":)" to "EMOJIsmile")
- **Replacing Usernames:** Replace @Usernames with word "USER". (eg: "@Ibrahim" to "USER")
- **Removing Non-Alphabets:** Replacing characters except Digits and Alphabets with a space.
- **Removing Consecutive letters:** 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")
- **Removing Short Words:** Words with length less than 2 are removed.
- **Removing Stopwords:** Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored  without sacrificing the meaning of the sentence. (eg: "the", "he", "have")
- **Lemmatizing:** Lemmatization is the process of converting a word to its base form. (e.g: “Great” to “Good”)


# Feature Extraction
For feature extraction, I used the TF-IDF technique.
TF-IDF indicates what the importance of the word is in order to 
understand the document or dataset.

TF-IDF Vectoriser converts a collection of raw documents to a matrix of 
TF-IDF features. The Vectoriser is usually trained on only the X_train dataset.

# Training and Evaluation
## Train/ Test split
80/ 20
## Logistic Regrssion
```
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
```
## Evaluation
```
y_pred = LRmodel.predict(X_test)
print("Classification Report:\n") 
print(classification_report(y_test, y_pred))
print("\n\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
```
![](https://i.ibb.co/L93GY1L/Screenshot-2021-10-10-003054.jpg)

# Saving the Model
```
file = open('vectoriser-ngram-(1,2).pickle','wb')
pickle.dump(vectoriser, file)
file.close()

file = open('Sentiment-LR.pickle','wb')
pickle.dump(LRmodel, file)
file.close()
```

# Using the Model
## 1. Load Model Function
```
def load_models(): 
    # Load the vectoriser.
    file = open('vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    
    # Load the LR Model.
    file = open('Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel
```

## 2. Predict Function
```
def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df
```
## 3. Use the model on some text
![](https://i.ibb.co/xHWjYmM/Screenshot-2021-10-10-003707.jpg)

# Real-Time Sentiment Analysis
## 1. Create the API
```
# Create the authentication object
authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Set the access token and access token secret
authenticate.set_access_token(access_token, access_secret)

# Create the API object
api = tweepy.API(authenticate, wait_on_rate_limit = True)
```

## 2. Extract Tweets
```
# Extract 100 tweets from BillGates Twitter account
posts = api.user_timeline(screen_name = 'BillGates', count = 30, lang = 'en', tweet_mode = 'extended')
```

## 3. Preprocess and Predict the Sentiment
```
text = [tweet.full_text for tweet in posts]
text = preprocess(text)

df2 = predict(vectoriser, LRmodel, text)
print(df2)
```
![](https://i.ibb.co/vB6hZFp/Screenshot-2021-10-10-004225.jpg)
# That's it! Thanks for Reading!

# Authors
- [@96ibman](https://www.github.com/96ibman)

# About Me
Ibrahim M. Nasser, a Software Engineer, Usability Analyst, 
and a Machine Learning Researcher.

# 🔗 Links
[![GS](https://img.shields.io/badge/-Google%20Scholar-blue)](https://scholar.google.com/citations?user=SSCOEdoAAAAJ&hl=en&authuser=2/)

[![linkedin](https://img.shields.io/badge/-Linked%20In-blue)](https://www.linkedin.com/in/ibrahimnasser96/)

[![Kaggle](https://img.shields.io/badge/-Kaggle-blue)](https://www.kaggle.com/ibrahim96/)


# Contact
96ibman@gmail.com

  