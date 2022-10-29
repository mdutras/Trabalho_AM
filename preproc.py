from re import search
from string import punctuation
import nltk
from unidecode import unidecode
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math
import string
import pandas as pd

def preProcessing(text, vocabulary):
    #nltk.download('stopwords')
    #nltk.download('rslp')
    #nltk.download('punkt')
    st = nltk.stem.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    #text = unidecode(text.lower())
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    text = text.translate(str.maketrans("","", string.digits))
    text = text.replace("\n", " ")
    text = nltk.word_tokenize(text)
    text = list(filter(lambda s : s not in stopwords, text))
    text = [st.stem(w) for w in text]
    vocabulary.update(text)
    return text

def createCaracVector(dataset, vocabulary):
    # Term Frequency (TF)
    TF = []
    for document in dataset:
        docTF = []
        tamDoc = len(document)
        for word in vocabulary:
            docTF.append(document.count(word) / tamDoc)
        TF.append(docTF)
    
    #print(TF)

    # Inverse Data Frequency (IDF)
    N = len(dataset)
    IDF = []
    for j in range(len(vocabulary)):
        #print(f"{list(vocabulary)[j]} : {sum([1 if TF[i][j] > 0 else 0 for i in range(len(TF))])}")
        IDF.append(math.log2(N/(sum([1 if TF[i][j] > 0 else 0 for i in range(len(TF))]))))
    #print(IDF)

    # TF-IDF
    TF_IDF =[[TF[i][j] * IDF[j] for j in range(len(vocabulary))] for i in range(N)]
    print(TF_IDF)

def main():
    print("Hi >:3")
    vocabulary = set()
    text = ["I wish I loved the Human Race;",
    "I wish I loved its silly face;",
    "I wish I liked the way it walks;",
    "I wish I liked the way it talks;",
    "And when I\'m introduced to one,",
    "I wish I thought \"What Jolly Fun!\""]
    #dataset = [preProcessing(text[i], vocabulary) for i in range(len(text))]
    #print(dataset)
    #createCaracVector(dataset, vocabulary)
    # Usando biblioteca pronta
    vectorizer = TfidfVectorizer()
    #texto = [" ".join(doc) for doc in dataset]
    cv = CountVectorizer()
    word_count = cv.fit_transform(text)
    print(word_count)
    
    #vectors = vectorizer.fit_transform(text)
    #feature_names = vectorizer.get_feature_names_out()
    #print(vocabulary)
    #print(feature_names)
    #dense = vectors.todense()
    #denselist = dense.tolist()
    #df = pd.DataFrame(denselist, columns=feature_names)
    #print(df)


if __name__ == "__main__":
    main()
