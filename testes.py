import nltk
import gensim
import sklearn
import re
import pandas as pd
import numpy as np
import matpltlib.pyplot as plt
import os
import json
from random import randint
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
    minDf = 5
    # Term Frequency
    countVec = CountVectorizer(min_df=minDf, tokenizer=customTokenizer)
    TF = countVec.fit_transform(docs)
    # Inverse Document Frequency
    tfidfTransf = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidfTransf.fit(TF)
    IDF = countVec.transform(docs)
    # TF-IDF
    TFIDF = tfidfTransf.transform(IDF)
    return TFIDF.todense().tolist()

def classifier(features, classes, classif, randNum):
    kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=randNum)
    measure = {
        "precision": 0,
        "accuracy": 0,
        "recall": 0,
        "micro f1": 0,
        "macro f1":0
    }
    for train, test in kf.split(features, classes):
        model = classif.fit(features[train], classes[train])
        prediction = np.array(model.predict(features[test]))
        measure["accuracy"] += accuracy_score(classes[test], prediction)
        measure["precision"] += precision_score(classes[test], prediction, average='weighted', zero_division=1)
        measure["recall"] += recall_score(classes[test], prediction, average='weighted', zero_division=1)
        measure["micro f1"] += f1_score(classes[test], prediction, average='micro', zero_division=1)
        measure["macro f1"] += f1_score(classes[test], prediction, average='macro', zero_division=1)
    for key in measure:
        measure[key] /= kf.get_n_splits()
    return measure


def main():
    results = {}
    for f in os.listdir("Datasets"):
        metrics = {}
        data = pd.read_csv("./Datasets/"+f)
        classes = data['genre']
        docs = data['synopsis']
        rs = randint(0,50)
        features = np.array(featureVector(docs))
        metrics["SVM (linear)"] = classifier(features, classes, SVC(kernel='linear', class_weight='balanced'), rs)
        metrics["SVM (sigmoid)"] = classifier(features, classes, SVC(kernel='sigmoid', class_weight='balanced'),rs)
        metrics["SVM (poly)"] = classifier(features, classes, SVC(kernel='poly', class_weight='balanced', degree=1), rs)
        metrics["SVM (rbf)"] = classifier(features, classes, SVC(kernel='rbf', class_weight='balanced'), rs)
        metrics["NB (gaussiano)"] = classifier(features, classes, GaussianNB(), rs)
        metrics["NB (Multinomial)"] = classifier(features, classes, MultinomialNB(), rs)
        metrics["NB (complemental)"] = classifier(features, classes, ComplementNB(), rs)
        metrics["KNN"] = classifier(features, classes, KNeighborsClassifier(), rs)
        metrics["Random Forest"] = classifier(features, classes, RandomForestClassifier(), rs)
        results[f] = metrics
    f = open("results.json", "w")
    f.write(json.dumps(results))
    f.close()
    metrics = ["precision", "accuracy", "recall", "micro f1", "macro f1"]
    for mtr in metrics:
        vals = [[] for i in range(len(results.keys()))]
        for clf in sorted(results[list(results.keys())[0]].keys()):
            i = 0
            for k in results.keys():
                vals[i].append(results[k][clf][mtr])
                i += 1
        x = sorted(sorted(results[list(results.keys())[0]].keys()))
        X_axis = np.arange(len(sorted(results[list(results.keys())[0]].keys())))
        plt.barh(X_axis + 0.2, vals[0], 0.2, label="TagMyBook")
        plt.barh(X_axis + 0.4, vals[1], 0.2, label="Book Genre Prediction")
        plt.title(f"{mtr}")
        plt.yticks(X_axis + 0.3, x, rotation=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{mtr.strip()}.png")
        plt.clf()


if __name__ == "__main__":
    main()
