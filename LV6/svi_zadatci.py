import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl)

data = pd.read_csv("Social_Network_Ads.csv")
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=10)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija:")
print(f"Tocnost train: {accuracy_score(y_train, y_train_p):.3f}")
print(f"Tocnost test: {accuracy_score(y_test, y_test_p):.3f}")

# 6.5.1 - KNN modeli
for K in [1, 5, 100]:
    KNN_model = KNeighborsClassifier(n_neighbors=K)
    KNN_model.fit(X_train_n, y_train)
    y_train_p = KNN_model.predict(X_train_n)
    y_test_p = KNN_model.predict(X_test_n)
    
    print(f"\nKNN (K={K}):")
    print(f"Tocnost train: {accuracy_score(y_train, y_train_p):.3f}")
    print(f"Tocnost test: {accuracy_score(y_test, y_test_p):.3f}")
    
    plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
    plt.title(f"KNN (K={K}) - Tocnost: {accuracy_score(y_train, y_train_p):.3f}")
    plt.show()

# 6.5.2
k_values = list(range(1, 31))
mean_scores = []
std_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_n, y_train, cv=5, scoring='accuracy')
    mean_scores.append(scores.mean())
    std_scores.append(scores.std())

optimal_k = k_values[np.argmax(mean_scores)]
print(f"\nOptimalna vrijednost K: {optimal_k}")

plt.figure()
plt.plot(k_values, mean_scores, 'o-')
plt.fill_between(k_values, 
                np.array(mean_scores) - np.array(std_scores),
                np.array(mean_scores) + np.array(std_scores),
                alpha=0.1)
plt.title("Unakrsna validacija za odabir K")
plt.xlabel("Broj susjeda (K)")
plt.ylabel("Točnost")
plt.xticks(k_values)
plt.grid()
plt.show()

# 6.5.3
C_values = [0.1, 1, 10, 100]
gamma_values = [0.01, 0.1, 1, 10]

for C in C_values:
    for gamma in gamma_values:
        svm_model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        svm_model.fit(X_train_n, y_train)
        y_test_p = svm_model.predict(X_test_n)
        
        print(f"\nSVM (C={C}, gamma={gamma}):")
        print(f"Tocnost test: {accuracy_score(y_test, y_test_p):.3f}")
        
        plot_decision_regions(X_train_n, y_train, classifier=svm_model)
        plt.title(f"SVM (C={C}, gamma={gamma})\nTocnost: {accuracy_score(y_train, svm_model.predict(X_train_n)):.3f}")
        plt.show()

# 6.5.4
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [0.01, 0.1, 1, 10]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel='rbf'))
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nNajbolji parametri:")
print(grid_search.best_params_)
print(f"Najbolja točnost: {grid_search.best_score_:.3f}")

# Vizualizacija najboljeg modela
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_n, y_train)
y_test_p = best_svm.predict(X_test_n)

print(f"\nNajbolji SVM model:")
print(f"Tocnost train: {accuracy_score(y_train, best_svm.predict(X_train_n)):.3f}")
print(f"Tocnost test: {accuracy_score(y_test, y_test_p):.3f}")

plot_decision_regions(X_train_n, y_train, classifier=best_svm)
plt.title(f"SVM (optimalni parametri)\nTocnost: {accuracy_score(y_train, best_svm.predict(X_train_n)):.3f}")
plt.show()