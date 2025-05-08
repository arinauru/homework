# переобучение присуще всем деревьям принятия решений
# ансамблевые методы. в основе идея обьединения нескольких (!) моделей для уменьшения эффекта переобучения
# это называется баггинг
# баггинг усредняет результаты -> оптимальной классификации
# ансамбль случайных дереевьев называется случайным лесом
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

iris = sns.load_dataset("iris")

print(iris.head())

sns.pairplot(iris, hue="species")
plt.show()

species_int = []
for r in iris.values:
    match r[4]:
        case "setosa":
            species_int.append(1)
        case "versicolor":
            species_int.append(2)
        case "virginica":
            species_int.append(3)

species_int_df = pd.DataFrame(species_int)
print(species_int_df.head())

data = iris[["sepal_length", "petal_length", "species"]]
data["species"] = species_int_df

data_setosa = data[data["species"] == 1]
data_versicolor = data[data["species"] == 2]
data_virginica = data[data["species"] == 3]

print(data_versicolor.shape)
print(data_virginica.shape)

data_versicolor_A = data_versicolor.iloc[:25, :]
data_versicolor_B = data_versicolor.iloc[25:, :]

print(data_versicolor_A)
print(data_versicolor_B)

data_virginica_A = data_virginica.iloc[:25, :]
data_virginica_B = data_virginica.iloc[25:, :]

data_df_A = pd.concat([data_virginica_A, data_versicolor_A], ignore_index=True)
data_df_B = pd.concat([data_virginica_B, data_versicolor_B], ignore_index=True)
data_df = pd.concat([data_virginica_A, data_versicolor_A], ignore_index=True)
data_df = pd.concat([data_virginica_B, data_versicolor_B], ignore_index=True)

x1_p = np.linspace(min(data["sepal_length"]), max(data["sepal_length"]), 100)
x2_p = np.linspace(min(data["petal_length"]), max(data["petal_length"]), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"])

fig, ax = plt.subplots(1, 3, sharex="col", sharey="row")

ax.scatter(data_setosa["sepal_length"], data_setosa["petal_length"])
ax.scatter(data_versicolor["sepal_length"], data_versicolor["petal_length"])
ax.scatter(data_virginica["sepal_length"], data_virginica["petal_length"])

ax[1].scatter(data_setosa["sepal_length"], data_setosa["petal_length"])
ax[1].scatter(data_versicolor["sepal_length"], data_versicolor["petal_length"])
ax[1].scatter(data_virginica["sepal_length"], data_virginica["petal_length"])

ax[2].scatter(data_setosa["sepal_length"], data_setosa["petal_length"])
ax[2].scatter(data_versicolor["sepal_length"], data_versicolor["petal_length"])
ax[2].scatter(data_virginica["sepal_length"], data_virginica["petal_length"])

md = 6

X = data_df["sepal_length", "petal_length"]
y = data_df["species"]
'''
# Bagging

model2=DecisionTreeClassifier(max_depth=md)
b=BaggingClassifier(model2, n_estimators=2, max_samples=0.5, random_state=1)
b.fit(X, y)

y_p2 = b.predict(X_p)
ax[1].contourf(X1_p, X2_p, y_p2.reshape(X1_p.shape), alpha=0.4, levels=2, cmap="rainbow", zorder=1)



max_depth = [1, 3, 5, 7]

X = data_df_A["sepal_length", "petal_length"]
y = data_df_A["species"]

j = 0
for md in max_depth:
    model = DecisionTreeClassifier(max_depth=md)
    model.fit()
    ax[0, j].scatter(data_virginica_A["sepal_length"], data_virginica_A["petal_length"])
    ax[0, j].scatter(data_versicolor_A["sepal_length"], data_versicolor_A["petal_length"])
    y_p = model.predict(X_p)
    ax[0, j].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, levels=2, cmap="rainbow", zorder=1)
    j += 1
'''

# Random Forest

model3 = RandomForestClassifier(n_estimators=2, max_samples=0.5, random_state=1)
model3.fit(X, y)

y_p3 = model3.predict(X_p)
ax[2].contourf(X1_p, X2_p, y_p3.reshape(X1_p.shape), alpha=0.4, levels=2, cmap="rainbow", zorder=1)

# регрессия с помощью случайных лесов

iris = sns.load_dataset("iris")

data = iris[["sepal_length", "petal_length", "species"]]

data_setosa = data[data["species"] == "setosa"]

x_p = pd.DataFrame(np.linspace(min(data_setosa["sepal_length"]), max(data_setosa["sepal_length"]), 100))

X = pd.DataFrame(data_setosa["sepal_length"], columns=["sepal_length"])
y = data["petal_length"]

model = RandomForestClassifier(n_estimators=20)
model.fit(X, y)

y_p = model.predict(x_p)

plt.scatter(data_setosa["sepal_length"], data_setosa["petal_length"])

plt.plit(x_p, y_p)

plt.show()

# Достоинства
# - простота и быстрота. распараллеливание процесса -> выигрыш во времени
# - вероятностная классификация
# - модель непараметрическая => хорошо работает с задачами, где другие модели могут оказаться недообученными
# Недостатки
# - сложно интерпретировать

# Метод главных компонент
# PCA ( principal component analysis) - алгоритм обученяи без учителя.
# PCF часто используют для понижения размерности

# задача машинного обучения без учителя состоит в выяснении зависимости между признаками
# В PCA выполняется качсетвенная оценка этой зависимости путем посика главных осей координат и их использования
# для описания набора данных

iris = sns.load_dataset("iris")

sns.pairplot(iris, hue="species")

data = iris[["petal_width", "petal_length", "species"]]
data_v = data[data["species"] == "versicolor"]

data_v=data_v.drop(columns=["species"])
print(data_v)

X = data_v["petal_width"]
Y = data_v["petal_length"]

plt.scatter(X, Y)

p=PCA(n_components=2)
p.fit(data_v)
X_p=p.transform(data_v)

print(data_v.shape)
print(X_p.shape)

X_p_new=p.inverse_transform(X_p)
print(X_p_new)

print(p.components_)
print(p.explained_variance_)
print(p.mean_)

plt.scatter(p.mean_[0], p.mean_[1])

plt.plot([p.mean_[0], p.mean_[0]+p.components_[0][0]],[p.mean_[1], p.mean_[1]+p.components_[0][1]])

plt.show()

# + простота интерпретации, эффективность в работе с многомерными данными
# - аномальные значения в данных оказывают сильное влияние

























