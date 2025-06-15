from sklearn.datasets import load_digits
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

digits = load_digits()
X, y = digits.data, digits.target

cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

models = {
'GNB': GaussianNB(),
'KNN': KNeighborsClassifier(),
'DT': DecisionTreeClassifier(random_state=42)
}
results_standard = {name: [] for name in models.keys()}

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)  # trenowanie na przeskalowanych danych treningowych
        y_pred = model.predict(X_test_scaled)  # predykcja na przeskalowanych danych testowych
        acc = accuracy_score(y_test, y_pred)  # obliczenie dokladnosci
        results_standard[name].append(acc)

print("\nZadanie 2")
print("\nStandard scaler")
for name, scores in results_standard.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"{name} {mean_score:.3f} ({std_score:.3f})")