import numpy as np
import pandas as pd

# # 1. Разобраться как использовать мультииндексные ключи в данном примере
index = [
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
]

population = [
    101,
    201,
    102,
    202,
    103,
    203,
]
pop = pd.Series(population, index=index)
pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }
)

# выборка данных для одного города и одного столбца
pop_df_1 = pop_df.loc['city_1', 'something']

# выборка данных для нескольких городов и нескольких столбцов
pop_df_1 = pop_df.loc[['city_1', 'city_3'], ['total', 'something']]

# выборка данных для нескольких городов и одного столбца
pop_df_1 = pop_df.loc[['city_1', 'city_3'], 'something']

# 2. Из получившихся данных выбрать данные по
# - 2020 году (для всех столбцов)
# - job_1 (для всех строк)
# - для city_1 и job_2

data = {('city_1', 2010): 100,
        ('city_1', 2020): 200,
        ('city_2', 2010): 1001,
        ('city_2', 2020): 2001}

s = pd.Series(data)
s.index.names = ['city', 'year']

index = pd.MultiIndex.from_product([['city_1', 'city_2'], [2010, 2020]],
                                   names=['city', 'year'])

columns = pd.MultiIndex.from_product([['person_1', 'person_2', 'person_3'],
                                      ['job_1', 'job_2']],
                                     names=['worker', 'job'])

rng = np.random.default_rng(1)
data = rng.random((4, 6))

data_df = pd.DataFrame(data, index=index, columns=columns)
print(data_df)

print(data_df.loc[(slice(None), 2020), :])

print(data_df.loc[:, (slice(None), 'job_1')])

print(data_df.loc['city_1', (slice(None), 'job_2')])

# 3. Взять за основу DataFrame со следующей структурой
# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3
# - все данные по первому городу и первым двум person-ам (с использование срезов)
#
# Приведите пример (самостоятельно) с использованием pd.IndexSlice

index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)

columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)
data = np.random.rand(4, 6)
df = pd.DataFrame(data, index=index, columns=columns)

print(df)

# 1. Все данные по person_1 и person_3
index = pd.IndexSlice
print(df.loc[:, index[['person_1', 'person_3'], :]])

# 2. Все данные по первому городу и первым двум person-ам
print(df.loc[index['city_1', :], index[['person_1', 'person_2'], :]])

# 3. Пример использования pd.IndexSlice для более сложного запроса
print(df.loc[index['city_2', 2020], index[:, 'job_1']])

# 4. Привести пример использования inner и outer джойнов для Series (данные примера скорее всего нужно изменить)
# ser1 = pd.Series(['a', 'b', 'c'], index=[1,2,3])
# ser2 = pd.Series(['b', 'c', 'f'], index=[4,5,6])

# print (pd.concat([ser1, ser2], join='outer'))
# print (pd.concat([ser1, ser2], join='inner'))

### Пример с пересекающимися индексами

ser1 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
ser2 = pd.Series(['b', 'c', 'f'], index=[2, 3, 4])

print(pd.concat([ser1, ser2], join='outer'))
print(pd.concat([ser1, ser2], join='inner'))
