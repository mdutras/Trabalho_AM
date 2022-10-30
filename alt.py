import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import gensim

def customPreprocessing(doc):
    stopWords = list(set(nltk.corpus.stopwords.words('english')) | set(gensim.parsing.preprocessing.STOPWORDS))
    st = nltk.stem.PorterStemmer()
    doc = doc.lower()
    doc = re.sub(r"[,.;@#?!&$'\"0-9]+", " ", doc)
    doc = nltk.word_tokenize(doc)
    doc = list(filter(lambda s : s not in stopWords, doc))
    doc = [st.stem(w) for w in doc
    return doc

def caracVector(docs):
    countVec = CountVectorizer(min_df=1, tokenizer=customPreprocessing)
    TF = countVec.fit_transform(docs)
    tfidfTransf = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidfTransf.fit(TF)
    IDF = countVec.transform(docs)
    TFIDF = tfidfTransf.transform(IDF)
    print(countVec.get_feature_names_out())
    print(pd.DataFrame(TFIDF.todense().tolist(), columns=countVec.get_feature_names_out()))
    return TFIDF.todense().tolist()

def main():
    docs = ["I wish I loved the Human Race;",
    "I wish I loved its silly face;",
    "I wish I liked the way it walks;",
    "I wish I liked the way it talks;",
    "And when I\'m introduced to one,",
    "I wish I thought \"What Jolly Fun!\""]
    #print(customTokenizer(text[4]))
    caracVector(docs)

if __name__ == "__main__":
    main()
