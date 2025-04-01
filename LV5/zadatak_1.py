import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                          random_state=213, n_clusters_per_class=1, class_sep=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)
plt.figure()
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', label='Class 0 (Train)', marker='o')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', label='Class 1 (Train)', marker='o')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='blue', label='Class 0 (Test)', marker='x')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='red', label='Class 1 (Test)', marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training and Test Data')
plt.legend()
plt.show()

# b)
model = LogisticRegression()
model.fit(X_train, y_train)

# c)
theta0 = model.intercept_[0]
theta1, theta2 = model.coef_[0]

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

plt.figure()
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', label='Class 0', marker='o')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', label='Class 1', marker='o')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary and Training Data')
plt.legend()
plt.show()

# d)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# e)
plt.figure()
correct = y_pred == y_test
incorrect = ~correct

plt.scatter(X_test[correct, 0], X_test[correct, 1], c='blue', label='Correct', marker='x')
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c='red', label='Incorrect', marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Test Set Classification Results')
plt.legend()
plt.show()
