from flask import Flask, jsonify
from flask import request
import pandas as pd
import GetOldTweets3 as got
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
s_stemmer = SnowballStemmer(language='english')
from sklearn.pipeline import Pipeline

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

app.config["DEBUG"] = True

def WS(income):
    text = []
    for i in  income:
        a= i.replace("_"," ")
        text.append(a)
    return text


def twittersearch(text):
    data = pd.DataFrame(columns=['username', 'to', 'text', 'retweets', 'favorites', 'replies', 'id',
       'permalink', 'author_id', 'date', 'formatted_date', 'hashtags',
       'mentions', 'geo', 'urls'])
    for i in text:
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(i)\
                                           .setMaxTweets(5)
        tweet = got.manager.TweetManager.getTweets(tweetCriteria)

        df = pd.DataFrame(t.__dict__ for t in tweet)

        data = data.append(df)
        
    data.drop_duplicates(subset ="text", inplace = True)
    
    return data

def cleantext(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #Remove @mentions
    text = re.sub(r'#', '',text)
    text = re.sub(r'https?:\/\/\S+','',text)#Removing # from tweets
    text = re.sub(r'  +', ' ', text) #removing extra spaces  
    return text

def cleantext2(data):
    raw_2 = []
    for i, j in data.itertuples():
        cl = cleantext(j)
        raw_2.append(cl)
    return raw_2

# loading the contraction dict
f = open('contraction.pkl', 'rb')
contraction = pickle.load(f)


#changing the contration words
def RC(df):
    for i, j in df.itertuples():
        for word in j.split():
            if word.lower() in contraction:
                m = j.replace(word, contraction[word.lower()])
                df.xs(i)["text"] = m
    return df

stopwords = pd.read_csv("stopwords.csv")
s = stopwords["0"]
sw = set(s)

#removing the search
def rese(income):
    for i in income:
        stops = word_tokenize(i)
        stops = [word for word in stops if word.isalpha()]
        stops = [word.lower() for word in stops]
        stops = ' '.join(stops)
        sw.add(i)

def cleaning(df):
    for i, j in df.itertuples():
        tokens = word_tokenize(j)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if not word in sw]
        #words = [lemmatizer.lemmatize(word) for word in tokens]
        words = [s_stemmer.stem(word) for word in tokens]
        sen = ' '.join(words)
        df.loc[i,"text"] = sen
    return df

model = pickle.load(open("model.sav","rb"))

def posi(y_pred):
    y = pd.DataFrame(y_pred, columns = ["result"] )
    n_values = y["result"].value_counts()
    pos = n_values[1]
    return pos


@app.route("/")
def home():
    return "<h1>Hi This is a Twitter Rating System<h1>"

@app.route("/predict")
def mst():
    income = request.args['key']
    income = income.split(",")
    income = WS(income)
    rese(income)
    data = twittersearch(income)
    raw = data[["text"]]
    raw_2 = cleantext2(raw)
    df = pd.DataFrame(raw_2, columns = ["text"])
    df = RC(df)
    df = cleaning(df)
    x_test = df["text"]
    y_pred = model.predict(x_test)
    pos = posi(y_pred)
    total = len(x_test)
    score = (pos / total)
    per = round(score*100,1)
    likage = f'{per}%'
    return str(f'<h1>{likage}<h1>')

if __name__ == '__main__':
    app.run(port = 5000,debug=True)
