from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

def main():
    docs =np.array(["I wish I loved the Human Race;",
    "I wish I loved its silly face;",
    "I wish I liked the way it walks;",
    "I wish I liked the way it talks;",
    "And when I\'m introduced to one,",
    "I wish I thought \"What Jolly Fun!\""])
    classes = np.array([0,1,2,1,0,2])
    kf = StratifiedKFold(n_splits=2,shuffle=True)
    print(kf.get_n_splits(docs))
    for train, test in kf.split(docs, classes):
        print(f"treino = {train} teste = {test}")



if __name__ == "__main__":
    main()
