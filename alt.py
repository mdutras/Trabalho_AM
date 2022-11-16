import nltk
import gensim
import sklearn
import re
import pandas as pd
import numpy as np
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

def customTokenizer(doc):
    stopWords = list(set(nltk.corpus.stopwords.words('english')) | set(gensim.parsing.preprocessing.STOPWORDS))
    st = PorterStemmer()
    doc = re.sub(r"[,.;@#?!&$'\"0-9]+", " ", doc.lower())
    doc = nltk.word_tokenize(doc)
    doc = list(filter(lambda s : s not in stopWords, doc))
    doc = [st.stem(w) for w in doc]
    return doc

def featureVector(docs, minDf=10):
    # Term Frequency
    print("Pré-processamento")
    countVec = CountVectorizer(min_df=minDf, tokenizer=customTokenizer)
    TF = countVec.fit_transform(docs)
    # Inverse Document Frequency
    tfidfTransf = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidfTransf.fit(TF)
    IDF = countVec.transform(docs)
    # TF-IDF
    print("TF-IDF")
    TFIDF = tfidfTransf.transform(IDF)
    return TFIDF.todense().tolist()

def classifier(features, classes, classif, extra={}):
    print(f"Classificador : {extra['classifierName']}")
    #docsTrain, docsTest, classesTrain, classesTest = train_test_split(features, classes, train_size=0.7, test_size=0.3,random_state=109)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=extra["randNum"])
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
    f = open("results.csv", "a")
    for key, value in measure.items():
        text = f"{extra['classifierName']}, {kf.get_n_splits()}, {extra['min_df']}, {extra['randNum']}, {key}, {value}, {len(set(classes[test]) - set(prediction))}\n"
        f.write(text)
    f.close()
    #print(measure)


def main():
    # docs = ["I wish I loved the Human Race;",
    # "I wish I loved its silly face;",
    # "I wish I liked the way it walks;",
    # "I wish I liked the way it talks;",
    # "And when I\'m introduced to one,",
    # "I wish I thought \"What Jolly Fun!\""]
    # classes = [0,1,2,1,0]
    minDf = 5
    data = pd.read_csv('data.csv')
    classes = np.array(data['genre'])
    docs = np.array(data['synopsis'])
    features = np.array(featureVector(docs, minDf))
    rs = randint(0, 42)
    #classifier(features, classes, SVC(kernel='linear', class_weight='balanced'), {"randNum":rs, "classifierName":"SVM (kernel='linear')", "min_df":minDf})
    #classifier(features, classes, SVC(kernel='sigmoid', class_weight='balanced'), {"randNum":rs, "classifierName":"SVM (kernel='sigmoid')", "min_df":minDf})
#   classifier(features, classes, SVC(kernel='poly', class_weight='balanced', degree=1), {"randNum":rs, "classifierName":"SVM (kernel='poly', degree=1)", "min_df":minDf})
    #classifier(features, classes, SVC(kernel='rbf', class_weight='balanced'), {"randNum":rs, "classifierName":"SVM (kernel='rbf')", "min_df":minDf})
#   classifier(features, classes, GaussianNB(), {"randNum":rs, "classifierName":"Gaussian NB", "min_df":minDf})
#   classifier(features, classes, MultinomialNB(), {"randNum":rs, "classifierName":"Multinomial NB", "min_df":minDf})
#   classifier(features, classes, ComplementNB(), {"randNum":rs, "classifierName":"Complement NB", "min_df":minDf})
#   classifier(features, classes, KNeighborsClassifier(), {"randNum":rs, "classifierName":"KNN", "min_df":minDf})
#   classifier(features, classes, RandomForestClassifier(), {"randNum":rs, "classifierName":"Random Forest", "min_df":minDf})
    classifier(features, classes, DecisionTreeClassifier(), {"randNum":rs, "classifierName":"Decision Tree (gini)", "min_df":minDf})
    classifier(features, classes, DecisionTreeClassifier(criterion='entropy'), {"randNum":rs, "classifierName":"Decision Tree (entropy)", "min_df":minDf})
    classifier(features, classes, DecisionTreeClassifier(criterion='log_loss'), {"randNum":rs, "classifierName":"Decision Tree (log_loss)", "min_df":minDf})


if __name__ == "__main__":
    main()
