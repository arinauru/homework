'''
#1
import plotly.express as px
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = px.data.iris()
df = df[df['species'].isin(['setosa', 'versicolor'])]
X = df[['sepal_length', 'sepal_width']]
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y_encoded)

x_min, x_max = X['sepal_length'].min() - 0.5, X['sepal_length'].max() + 0.5
y_min, y_max = X['sepal_width'].min() - 0.5, X['sepal_width'].max() + 0.5
x_m, y_m = np.meshgrid(np.linspace(x_min, x_max, 200),
                       np.linspace(y_min, y_max, 200))
Z = svm.decision_function(np.c_[x_m.ravel(), y_m.ravel()]).reshape(x_m.shape)

fig = px.scatter(
    df,
    x='sepal_length',
    y='sepal_width',
    color='species',
    color_discrete_map={
        'setosa': 'blue',
        'versicolor': 'red'
    }
)

fig.add_contour(
    x=x_m[0],
    y=y_m[:, 0],
    z=Z,
    contours=dict(start=0, end=0, size=0.1, coloring="lines"),
    line_width=3,
    colorscale=[[0, 'black'], [1, 'black']],
    showscale=False,
    name="Граница решения"
)

# Заливка областей
fig.add_contour(
    x=x_m[0],
    y=y_m[:, 0],
    z=(Z > 0).astype(int),
    contours_coloring='fill',
    colorscale=[[0, 'rgba(255,0,0,0.1)'], [1, 'rgba(0,0,255,0.1)']],
    showscale=False,
    line_width=0,
    name="Области классов"
)

fig.update_layout(
    title='SVM: Граница решения',
    legend_title_text='Классы',
    xaxis_title="Длина чашелистика",
    yaxis_title="Ширина чашелистика",
    legend=dict(
        itemsizing='constant',
        title_font_size=14,
        font_size=12
    )
)

fig.show()

#2
import plotly.express as px
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

df = px.data.iris()
df = df[df['species'].isin(['setosa', 'versicolor'])]

X_full = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_full)

svm_pca = SVC(kernel='linear', C=0.1)
svm_pca.fit(X_pca, y_encoded)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
x_m, y_m = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = svm_pca.decision_function(np.c_[x_m.ravel(), y_m.ravel()]).reshape(x_m.shape)

fig = px.scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    color=df['species'],
    color_discrete_map={'setosa': 'blue', 'versicolor': 'red'}
)

fig.add_contour(
    x=x_m[0],
    y=y_m[:, 0],
    z=Z,
    contours=dict(start=0, end=0, coloring="lines"),
    line_width=3,
    colorscale=[[0, 'black'], [1, 'black']],
    showscale=False
)

fig.update_layout(
    title='PCA + SVM: Гладкая граница',
    xaxis_title="Главная компонента 1",
    yaxis_title="Главная компонента 2",
    legend_title="Классы"
)

fig.show()

'''
# 3
import plotly.express as px
from sklearn.cluster import KMeans

df = px.data.iris()
df = df[df['species'].isin(['setosa', 'versicolor'])]

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(df[['sepal_length', 'sepal_width', 'petal_width']])

fig = px.scatter_3d(
    df,
    x='sepal_length',
    y='sepal_width',
    z='petal_width',
    color='species',
    labels={
        "sepal_length": "Длина чашелистика",
        "sepal_width": "Ширина чашелистика",
        "petal_width": "Ширина лепестка"
    }
)

fig.update_layout(
    legend_title_text='Классы',
    font=dict(size=12),
    scene=dict(
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        zaxis_title_font=dict(size=14)
    )
)

fig.show()
