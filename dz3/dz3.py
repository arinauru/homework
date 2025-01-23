import numpy as np
import pandas as pd

# 1. Привести различные способы создания объектов типа Series
# Для создания Series можно использовать
# - списки Python или массивы NumPy
# - скалярные значение
# - словари

# Создание Series из списка
data_list = [1, 2, 3, 4]
series_from_list = pd.Series(data_list)
print(series_from_list)

# Создание Series из массива NumPy
data_array = np.array([1, 2, 3, 4])
series_from_array = pd.Series(data_array)
print(series_from_array)

# Создание Series из скалярного значения
scalar_value = 5
index = ['a', 'b', 'c', 'd']
series_from_scalar = pd.Series(scalar_value, index=index)
print(series_from_scalar)

# Создание Series из словаря
data_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
series_from_dict = pd.Series(data_dict)
print(series_from_dict)

# 2. Привести различные способы создания объектов типа DataFrame
# DataFrame. Способы создания
# - через объекты Series
# - списки словарей
# - словари объектов Series
# - двумерный массив NumPy
# - структурированный массив Numpy

# Создание объектов Series
series1 = pd.Series([1, 2, 3])
series2 = pd.Series(['a', 'b', 'c'])
series3 = pd.Series([True, False, True])

# Создание DataFrame из объектов Series
df_from_series = pd.DataFrame({
    'c1': series1,
    'c2': series2,
    'c3': series3
})

print(df_from_series)

# Создание DataFrame из списка словарей
data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
df_from_list_of_dicts = pd.DataFrame(data)
print(df_from_list_of_dicts)

# Создание DataFrame из словаря Series
data = {
    'col1': pd.Series([1, 2, 3]),
    'col2': pd.Series(['a', 'b', 'c'])
}
df_from_dict_of_series = pd.DataFrame(data)
print(df_from_dict_of_series)

# Создание двумерного массива NumPy
data = np.array([[1, 2], [3, 4], [5, 6]])

# Создание DataFrame из массива NumPy
df_from_numpy_array = pd.DataFrame(data, columns=['c1', 'c2'], index=['r1', 'r2', 'r3'])
print(df_from_numpy_array)

# Создание структурированного массива NumPy
data = np.array([
    (1, 'Kate', 15.5),
    (2, 'Bob', 26.2),
    (3, 'Alice', 13.0)
], dtype=[('id', 'i4'), ('name', 'U10'), ('age', 'f4')])

# Создание DataFrame из структурированного массива
df_from_structured_array = pd.DataFrame(data)
print(df_from_structured_array)

# 3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1

# Создание двух объектов Series с разными индексами
series1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
series2 = pd.Series([40, 50, 60], index=['b', 'c', 'd'])

print(series1.add(series2, fill_value=1))

# 4. Переписать пример с транслирование для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ

rng = np.random.default_rng()

dfA = pd.DataFrame(rng.integers(0, 10, (2, 2)), columns=['a', 'b'])
dfB = pd.DataFrame(rng.integers(0, 10, (3, 3)), columns=['a', 'b', 'c'])

print(dfA)
print(dfB)
print(dfA.sub(dfB['a'], axis=0))

# 5. На примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()

data = {
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [np.nan, 2, np.nan, 4, np.nan],
    'C': [6, np.nan, np.nan, 9, 10]
}
df = pd.DataFrame(data)
print(df)

# Заполнение пропусков методом ffill()
df_ffilled = df.ffill()
print(df_ffilled)

# Заполнение пропусков методом bfill()
df_bfilled = df.bfill()
print(df_bfilled)

# Комбинирование ffill() и bfill()
df_filled = df.ffill().bfill()
print(df_filled)