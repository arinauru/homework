import numpy as np

'''
a = np.ones((3, 2))
b = np.arange(3)

# 2 (3, 2) -> (3, 2) -> (3, 2)
# 1 (3, ) -> (1, 3) -> (3, 3)

c = a + b'''

## 1. Что надо изменить в последнем примере, чтобы он заработал без ошибок (транслирование)?

## Чтобы исправить ошибку, нужно изменить форму массива b на совместимую с a. Например, можно изменить его форму на (3, 1):

a = np.ones((3, 2))
b = np.arange(3).reshape((3, 1))

c = a + b
print(c, c.shape)

## 2. Пример для y. Вычислить количество элементов (по обоим размерностям), значения которых больше 3 и меньше 9

y = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

mask = (y > 3) & (y < 9)

print(mask)

count_total = np.sum(mask)
count_axis0 = np.sum(mask, axis=0)  # по оси 0 (строки)
count_axis1 = np.sum(mask, axis=1)  # по оси 1 (столбцы)

print(count_total)
print(count_axis0)
print(count_axis1)

print(np.where(mask, y, 0))
