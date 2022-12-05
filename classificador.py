import nltk
import gensim
import sklearn
import re
import pandas as pd
import numpy as np
import os
import json
from random import randint
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.svm import SVC
nltk.download('stopwords')
nltk.download('punkt')

def customTokenizer(doc):
    stopWords = list(set(nltk.corpus.stopwords.words('english')) | set(gensim.parsing.preprocessing.STOPWORDS))
    st = PorterStemmer()
    doc = re.sub(r"[,.;@#?!&$'\"0-9]+", " ", doc.lower())
    doc = nltk.word_tokenize(doc)
    doc = list(filter(lambda s : s not in stopWords, doc))
    doc = [st.stem(w) for w in doc]
    return doc

def featureVector(docs):
    # Term Frequency
    countVec = CountVectorizer(min_df=5, tokenizer=customTokenizer)
    TF = countVec.fit_transform(docs)
    # Inverse Document Frequency
    tfidfTransf = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidfTransf.fit(TF)
    IDF = countVec.transform(docs)
    # TF-IDF
    TFIDF = tfidfTransf.transform(IDF)
    return TFIDF.todense().tolist()

def main():
    synopsisGiven = input("Insert the synopsis: ")
    classes = []
    docs = []
    for f in os.listdir("\Datasets"):
        data = pd.read_csv(f)
        classes += data['genre']
        docs += data['synopsis']
    features = np.array(featureVector(docs))
    svm = SVC(kernel='sigmoid', class_weight='balanced')
    model = svm.fit(features[train], classes[train])
    prediction = model.predict(synopsisGiven)
    print(f"The synopsis belongs to a book whose genre is {prediction}")

if __name__ == "__main__":
    main()
