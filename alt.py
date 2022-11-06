import nltk
import re
import gensim
import sklearn
import pandas as pd
import numpy as np
from random import randint
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB

def customTokenizer(doc):
    stopWords = list(set(nltk.corpus.stopwords.words('english')) | set(gensim.parsing.preprocessing.STOPWORDS))
    st = nltk.stem.PorterStemmer()
    doc = re.sub(r"[,.;@#?!&$'\"0-9]+", " ", doc.lower())
    doc = nltk.word_tokenize(doc)
    doc = list(filter(lambda s : s not in stopWords, doc))
    doc = [st.stem(w) for w in doc]
    return doc

def featureVector(docs):
    # Term Frequency
    print("Pré-processamento")
    countVec = CountVectorizer(min_df=10, tokenizer=customTokenizer)
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
    print("Classificador")
    #docsTrain, docsTest, classesTrain, classesTest = train_test_split(features, classes, train_size=0.7, test_size=0.3,random_state=109)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=extra["randNum"])
    measure = {
        "precision": 0,
        "accuracy": 0,
        "recall": 0,
        "f1": 0
    }
    for train, test in kf.split(features, classes):
        model = classif.fit(features[train], classes[train])
        prediction = np.array(model.predict(features[test]))
        if set(classes[test]) - set(prediction):
            print(set(classes[test]) - set(prediction))
        measure["accuracy"] += accuracy_score(classes[test], prediction)
        measure["precision"] += precision_score(classes[test], prediction, average='weighted')
        measure["recall"] += recall_score(classes[test], prediction, average='weighted')
        measure["f1"] += f1_score(classes[test], prediction, average='weighted')
        #print(f"--- {extra["classificador"]}\nAccuracy = {accuracy}\nPrecision = {precision}\nRecall = {recall}\nF1 = {f1}")
    for key in measure:
        measure[key] /= kf.get_n_splits()
    text = f"--- {extra['classifierName']} {extra['randNum']} ---\n- Accuracy = {measure['accuracy']}\n- Precision = {measure['precision']}\n- Recall = {measure['recall']}\n- F1 = {measure['f1']}\n\n"
    f = open("resultados.txt", "a")
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
    data = pd.read_csv('data.csv')
    classes = np.array(data['genre'])
    docs = np.array(data['synopsis'])
    features = np.array(featureVector(docs))
    for i in range(5):
        rs = randint(0, 42)
        classifier(features, classes, sklearn.svm.SVC(kernel='linear'), {"randNum":rs, "classifierName":"SVM"})

if __name__ == "__main__":
    main()
