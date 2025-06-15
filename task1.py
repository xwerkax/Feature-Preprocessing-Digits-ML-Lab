from sklearn.datasets import load_digits
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


digits = load_digits()
X, y = digits.data, digits.target

cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

models = {
'GNB': GaussianNB(),
'KNN': KNeighborsClassifier(),
'DT': DecisionTreeClassifier(random_state=42)
}
print("Zadanie 1")
print("\nBase dataset")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    mean_score = np.mean(scores)  # srednia
    std_score = np.std(scores)    # odchylenie stand.
    print(f"{name} {mean_score:.3f} ({std_score:.3f})")