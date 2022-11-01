import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import gensim
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score

def customTokenizer(doc):
    stopWords = list(set(nltk.corpus.stopwords.words('english')) | set(gensim.parsing.preprocessing.STOPWORDS))
    st = nltk.stem.PorterStemmer()
    doc = doc.lower()
    doc = re.sub(r"[,.;@#?!&$'\"0-9]+", " ", doc)
    doc = nltk.word_tokenize(doc)
    doc = list(filter(lambda s : s not in stopWords, doc))
    doc = [st.stem(w) for w in doc]
    return doc

def featureVector(docs):
    # Term Frequency
    countVec = CountVectorizer(min_df=1, tokenizer=customTokenizer)
    TF = countVec.fit_transform(docs)
    # Inverse Document Frequency
    tfidfTransf = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidfTransf.fit(TF)
    IDF = countVec.transform(docs)
    # TF-IDF
    TFIDF = tfidfTransf.transform(IDF)
    #print(countVec.get_feature_names_out())
    #print(pd.DataFrame(TFIDF.todense().tolist(), columns=countVec.get_feature_names_out()))
    #print(TFIDF.todense().tolist())
    return TFIDF.todense().tolist()

def SVMclass(features, classes, kernel):
    xTrain, xTest, yTrain, yTest = train_test_split(features, classes, train_size=0.7, test_size=0.3,random_state=109)
    classifier = sklearn.svm.SVC(kernel=kernel).fit(xTrain, yTrain)
    prediction = classifier.predict(xTest)
    accuracy = accuracy_score(yTest, prediction)
    f1 = f1_score(yTest, prediction, average='weighted')
    print(f"--- {kernel}\nAccuracy = {accuracy}\nF1 = {f1}")

def main():
    # docs = ["I wish I loved the Human Race;",
    # "I wish I loved its silly face;",
    # "I wish I liked the way it walks;",
    # "I wish I liked the way it talks;",
    # "And when I\'m introduced to one,",
    # "I wish I thought \"What Jolly Fun!\""]
    # classes = [0,1,2,1,0]
    data = pd.read_csv('data.csv')
    classes = data['genre']
    docs = data['synopsis']
    features = featureVector(docs)
    SVMclass(features, classes, "linear")
    SVMclass(features, classes, "poly")
    SVMclass(features, classes, "rbf")
    SVMclass(features, classes, "sigmoid")
    SVMclass(features, classes, "precomputed")

if __name__ == "__main__":
    main()
