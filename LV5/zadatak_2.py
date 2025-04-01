import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

labels = {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor='w',
                    label=labels[cl])

df = pd.read_csv("penguins.csv")

df = df.drop(columns=['sex'])
df.dropna(axis=0, inplace=True)

df['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}, inplace=True)

output_variable = ['species']
input_variables = ['bill_length_mm', 'flipper_length_mm']
X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# a)
train_counts = np.unique(y_train, return_counts=True)
test_counts = np.unique(y_test, return_counts=True)

plt.figure()
plt.bar([labels[i] for i in train_counts[0]], train_counts[1], alpha=0.5, label='Train')
plt.bar([labels[i] for i in test_counts[0]], test_counts[1], alpha=0.5, label='Test')
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.legend()
plt.show()

# b)
model = LogisticRegression()
model.fit(X_train, y_train)

# c)
print("Intercepts:", model.intercept_)
print("Coefficients:", model.coef_)

# d)
plot_decision_regions(X_train, y_train, classifier=model)
plt.xlabel('Bill length (mm)')
plt.ylabel('Flipper length (mm)')
plt.title('Decision Regions')
plt.show()

# e)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[labels[i] for i in range(3)])
disp.plot()
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=[labels[i] for i in range(3)]))

# f)
input_variables_ext = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X_ext = df[input_variables_ext].to_numpy()
y_ext = df[output_variable].to_numpy().ravel()

X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_ext, y_ext, test_size=0.2, random_state=123)

model_ext = LogisticRegression(max_iter=1000)
model_ext.fit(X_train_ext, y_train_ext)

y_pred_ext = model_ext.predict(X_test_ext)
print("\nClassification Report with Additional Features:")
print(classification_report(y_test_ext, y_pred_ext, target_names=[labels[i] for i in range(3)]))
