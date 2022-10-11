from re import search
from string import punctuation
import nltk
from unidecode import unidecode
import re

def preProcessing(text):
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('portuguese')
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    text = unidecode(text)
    text = text.replace("\n", " ")
    text = text.split(" ")
    text = list(filter(lambda s : s not in stopwords, text))
    vocabulary = set(text)
    print(text, len(text))
    print(len(vocabulary))


def main():
    print("Hi >:3")
    text = """Único romance da escritora inglesa Emily Bronte, O morro dos ventos uivantes retrata uma trágica historia de amor e obsessão em que os personagens principais são a obstinada e geniosa Catherine Earnshaw e seu irmão adotivo, Heathcliff. Grosseiro, humilhado e rejeitado, ele guarda apenas rancor no coração, mas tem com Catherine um relacionamento marcado por amor e, ao mesmo tempo, ódio."""
    preProcessing(text)

if __name__ == "__main__":
    main()
