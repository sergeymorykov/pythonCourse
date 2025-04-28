import numpy as np
import matplotlib.pyplot as plt
from utils import gradient_descent, mean_squared_error

# Загрузка данных
data = np.genfromtxt('data/data.csv', delimiter=',', skip_header=1)
mileage = data[:, 0]
price = data[:, 1]

# Нормализация данных (опционально, но улучшает сходимость)
norm_mileage = (mileage - np.mean(mileage)) / np.std(mileage)
norm_price = (price - np.mean(price)) / np.std(price)

#скорость обучения
learn_h_rate = 0.1
#количество итераций
number_of_iterations = 1000

# Обучение модели
norm_theta0, norm_theta1 = gradient_descent(norm_mileage, norm_price, learn_h_rate, number_of_iterations)

theta0 = norm_theta0 * np.std(price) + np.mean(price) - norm_theta1 * np.std(price) * np.mean(mileage) / np.std(mileage)
theta1 = norm_theta1 * np.std(price) / np.std(mileage)

# Сохранение параметров
with open('result/theta_values.txt', 'w') as f:
    f.write(f"{theta0},{theta1}")

# Вычисление среднеквадратичной ошибки
mse = mean_squared_error(norm_mileage, norm_price, norm_theta0, norm_theta1)
print(f"Среднеквадратичная ошибка: {mse}")

# Визуализация
plt.scatter(mileage, price, label='Данные')
plt.plot(mileage, theta0 + theta1 * mileage, 'r-', label='Линейная регрессия')
plt.xlabel('Пробег, км')
plt.ylabel('Цена')
plt.legend()
plt.show()

