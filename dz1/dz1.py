import array

import numpy as np
import sys

## 1. Какие еще существуют коды типов?
'''
Числовые типы:
int — целые числа (например, 12)
float — числа с плавающей точкой (например, 3.14)
complex — комплексные числа (например, 1 + 2i)
Строки:
str — строки (например, "hello")
Булевы значения:
bool — логические значения (True или False)
Последовательности:
list — списки (например, [1, 2, 3])
tuple — кортежи (например, (1, 2, 3))
range — диапазоны (например, range(10))
Множества:
set — множества (например, {1, 2, 3})
Словари:
dict — словари (например, {'a': 1, 'b': 2})
Битовые типы:
bytes — неизменяемые последовательности байт (например, b'hello')
bytearray — изменяемые последовательности байт
'''

## 2. Напишите код, подобный приведенному выше, но с другим типом

# list
x = [1, 2, 3]
print(type(x))
print(sys.getsizeof(x))

# tuple
x = (1, 2, 3)
print(type(x))
print(sys.getsizeof(x))

# set
x = {1, 2, 3}
print(type(x))
print(sys.getsizeof(x))

# dict
x = {'a': 1, 'b': 2}
print(type(x))
print(sys.getsizeof(x))

## 3. Напишите код для создания массива с 5 значениями, располагающимися через равные интервалы в диапазоне от 0 до 1

array = np.linspace(0, 1, 5)
print(array)

## 4. Напишите код для создания массива с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1

array = np.random.uniform(0, 1, 5)
print(array)

## 5. Напишите код для создания массива с 5 нормально распределенными случайными значениями с мат. ожиданием = 0 и дисперсией 1

array = np.random.normal(loc=0, scale=1, size=5)
print(array)

## 6. Напишите код для создания массива с 5 случайными целыми числами в от [0, 10)

array = np.random.randint(0, 10, size=5)
print(array)

## 7. Написать код для создания срезов массива 3 на 4
## - первые две строки и три столбца
## - первые три строки и второй столбец
## - все строки и столбцы в обратном порядке
## - второй столбец
## - третья строка

array = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print(array)

slice_1 = array[:2, :3]
print(slice_1)

slice_2 = array[:3, 1]
print(slice_2)

slice_3 = array[::-1, ::-1]
print(slice_3)

slice_4 = array[:, 1]
print(slice_4)

slice_5 = array[2, :]
print(slice_5)

## 8. Продемонстрируйте, как сделать срез-копию

array = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print(array)

slice = array[:2, :3]
print(slice)

slice_copy = array[:2, :3].copy()
print(slice_copy)

## 9. Продемонстрируйте использование newaxis для получения вектора-столбца и вектора-строки

array = np.array([1, 2, 3, 4])

column_vector = array[:, np.newaxis]
row_vector = array[np.newaxis, :]

print(column_vector)
print(row_vector)

## 10. Разберитесь, как работает метод dstack
'''
Метод numpy.dstack используется для объединения массивов вдоль третьей оси (оси глубины).
Это особенно полезно, когда нужно объединить несколько двумерных массивов (матриц) в один 
трехмерный массив, где каждая матрица становится "слоем" вдоль новой оси.'''
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

print(np.dstack((array1, array2)))

## 11. Разберитесь, как работают методы split, vsplit, hsplit, dsplit
'''
split-разделяет массив вдоль указанной оси
vsplit-Разделяет массив вертикально (по строкам)
hsplit-Разделяет массив горизонтально (по столбцам)
dsplit-Разделяет массив по глубине (вдоль третьей оси)
'''
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

for i in np.split(array, 3, axis=0):
    print(i)

for i in np.vsplit(array, 3):
    print(i)

for i in np.hsplit(array, 3):
    print(i)

array2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for i in np.dsplit(array2, 2):
    print(i)

## 12. Привести пример использования всех универсальных функций, которые я привел

x = np.array(10)
print(x)

print(x * 2 + 1)
print(np.add(x, 1))
print(np.multiply(x, 2))
print(np.subtract(x, 1))
print(np.divide(x, 2))
print(np.floor_divide(x, 3))
print(np.power(x, 2))
print(np.mod(x, 3))
