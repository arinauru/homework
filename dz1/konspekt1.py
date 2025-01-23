import array

import numpy as np
import sys

# Типы данных питон

x = 1
print(type(x))
print(sys.getsizeof(x))

x = 'hello'
print(type(x))

x = True
print(type(x))

l1 = list([])
print(sys.getsizeof(l1))

l2 = list([1, 2, 3])
print(sys.getsizeof(l2))

l3 = list([4, "2", True])
print(sys.getsizeof(l3))

a1 = array.array('i', [1, 2, 3])
print(sys.getsizeof(a1))
print(type(a1))

## 1. Какие еще существуют коды типов?

## 2. Напишите код, подобный приведенному выше, но с другим типом

a = np.array(([1, 2, 3, 4, 5]))
print(type(a), a)

a = np.array(([1.23, 2, 3, 4, 5]))
print(type(a), a)

a = np.array([range(i, i + 3) for i in [2, 4, 6]])
print(type(a), a)

a = np.zeros(10, dtype=int)
print(type(a), a)

print(np.ones((3, 5), dtype=float))

print(np.full((4, 5), 3.14))

print(np.arange((0, 20, 2)))

print(np.eye(4))

## 3. Напишите код для создания массива с 5 значениями, располагающимися через равные интервалы в диапазоне от 0 до 1

## 4. Напишите код для создания массива с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1

## 5. Напишите код для создания массива с 5 нормально распределенными случайными значениями с мат. ожиданием = 0 и дисперсией 1

## 6. Напишите код для создания массива с 5 случайнвми целыми числами в от [0, 10)

### Массивы

np.random.seed(1)

x1 = np.random.randint(10, size=3)
x2 = np.random.randint(10, size=(3, 2))
x3 = np.random.randint(10, size=(3, 2, 1))

print(x1)
print(x2)
print(x3)

print(x1.ndim, x1.shape, x1.size)
print(x2.ndim, x2.shape, x2.size)
print(x3.ndim, x3.shape, x3.size)

# Индекс (с 0)

a = np.array([1, 2, 3, 4, 5])
print(a[0])
print(a[-2])

a[1] = 20
print(a)

a = np.array([[1, 2], [3, 4]])
print(a)

print(a[0, 0])
print(a[-1, -1])

a[1, 0] = 100
print(a)

a = np.array([1, 2, 3, 4, 5])
b = np.array([1.0, 2, 3, 4])

print(a)
print(b)

a[0] = 10
print(a)

a[0] = 10.23
print(a)

# Срез [s:f:st] [1:shape:1]

a = np.array([1, 2, 3, 4, 5, 6])

print(a[:3])
print(a[3:])
print(a[1:5])
print(a[1:-1])
print(a[1::2])

print(a[::-1])

## 7. Написать код для создания срезов массива 3 на 4
## - первые две строки и три столбца
## - первые три строки и второй столбец
## - все строки и столбцы в обратном порядке
## - второй столбец
## - третья строка

a = np.array([1, 2, 3, 4, 5, 6])

b = a[:3]
print(b)

b[0] = 100
print(a)

## 8. Продемонстрируйте, как сделать срез-копию

a = np.array(1, 13)

print(a)

print(a.reshape(2, 6))
print(a.reshape(3, 4))

## 9. Продемонстрируйте использование newaxis для получения вектора-столбца и вектора-строки

x = np.array(([1, 2, 3]))
y = np.array(([4, 5]))
z = np.array(([6]))

print(np.concatenate([x, y, z]))

x = np.array(([1, 2, 3]))
y = np.array(([4, 5, 6]))

r1 = np.vstack([x, y])
print(r1)

print(np.hstack([r1, r1]))

## 10. Разберитесь, как работает метод dstack

## 11. Разберитесь, как работают методы split, vsplit, hsplit, dsplit

### Вычисления с массивами
# Векторизирвоанная операция

x = np.array(10)
print(x)

print(x * 2 + 1)

# Универсальные функции
print(np.add(np.multiply((x, 2), 1)))

# - / // ** %


## 12. Привести пример использования всех универсальных функций, которые я привел

## np.abs, sin/cos/tan, exp, log

x = np.array(5)

y = np.zeros(10)
print(np.multiply(x, 10, out=y[::2]))

print(y)

x = np.array(5)

print(x)

print(np.add.reduce((x)))
print(np.add.accumulate((x)))

x = np.arange(1, 10)
print(np.add.outer(x, x))
print(np.multiply.outer(x, x))
