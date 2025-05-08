import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

iris = sns.load_dataset("iris")
data = iris.loc[(iris["species"] == "setosa") | (iris["species"] == "versicolor"),
                ["sepal_length", "petal_length", "species"]].copy()

le = LabelEncoder()
scaler = StandardScaler()
data["species_encoded"] = le.fit_transform(data["species"])
X = scaler.fit_transform(data[["sepal_length", "petal_length"]])
X = pd.DataFrame(X, columns=["sepal_length", "petal_length"])
Y = data["species_encoded"]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for species in le.classes_:
    subset = data[data["species"] == species]
    plt.scatter(subset["sepal_length"], subset["petal_length"], label=species)
plt.title("Original Data")
plt.legend()

model = SVC(kernel="linear", C=1e15)
model.fit(X, Y)

x1_p = np.linspace(X["sepal_length"].min()-0.5, X["sepal_length"].max()+0.5, 100)
x2_p = np.linspace(X["petal_length"].min()-0.5, X["petal_length"].max()+0.5, 100)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
                 columns=["sepal_length", "petal_length"])

y_p = model.predict(X_p)
plt.contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.1, levels=1)
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=150, facecolor="none", edgecolor="black", linewidth=2, label='Support Vectors')

plt.subplot(1, 2, 2)
X_reduced = pd.DataFrame(model.support_vectors_, columns=X.columns)
Y_reduced = Y.iloc[model.support_]

model_reduced = SVC(kernel="linear", C=1e15)
model_reduced.fit(X_reduced, Y_reduced)

y_p_reduced = model_reduced.predict(X_p)
plt.contourf(X1_p, X2_p, y_p_reduced.reshape(X1_p.shape), alpha=0.1, levels=1)
plt.scatter(X_reduced["sepal_length"], X_reduced["petal_length"],
            s=100, c='red', edgecolor='black', label='Support Vectors Only')
plt.title("Reduced Model")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Совпадение предсказаний: {np.all(y_p == y_p_reduced)}")
print(f"Коэффициенты:\nПолная: {model.coef_[0]}\nСокращенная: {model_reduced.coef_[0]}")
print(f"Опорные векторы:\nПолная: {model.support_vectors_}\nСокращенная: {model_reduced.support_vectors_}")

plt.show()