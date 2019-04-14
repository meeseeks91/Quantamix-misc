# Store url
url = 'https://www.quantamixsolutions.com/blog/blogn/17'
"""Import `requests`
Make the request and check object type"""
import numpy as np
import pandas
import requests

from bs4 import BeautifulSoup
r = requests.get(url)
soup = BeautifulSoup(r.content)
souptext = soup.get_text()

import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter


# Create tokenizer
# Create tokens
tokenizer = RegexpTokenizer('\w+')
tokens = tokenizer.tokenize(souptext)

words = []
for word in tokens:
    words.append(word.lower())

sw = nltk.corpus.stopwords.words('english')
# Initialize new list
words_ns = []

# Add to words_ns all words that are in words but not in sw
for word in words:
    if word not in sw:
        words_ns.append(word)


#create nltk text
text = nltk.Text(words_ns)

# remove punctuation, count no stop raw words
stops= set(stopwords.words("english"))
nonPunct = re.compile('.*[A-Za-z].*')
raw_words = [w for w in text if nonPunct.match(w)]
raw_word_count = Counter(raw_words)
#corpus = no stop word
corpus = [w for w in raw_words if w.lower() not in stops]
corpus_count = Counter(corpus)




from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_df=0.8,stop_words=stops, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)


#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                       reverse=True)
    return words_freq[:n]
#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=20)
top_df = pandas.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

print (top_words)
#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
