import numpy as np
import matplotlib.pyplot as plt
from utils import gradient_descent

# Загрузка данных
data = np.genfromtxt('data/data.csv', delimiter=',', skip_header=1)
mileage = data[:, 0]
price = data[:, 1]

# Нормализация данных (опционально, но улучшает сходимость)
norm_mileage = (mileage - np.mean(mileage)) / np.std(mileage)
norm_price = (price - np.mean(price)) / np.std(price)

#скорость обучения
learning_rate = 0.01
#количество итераций
epochs = 10000

# Обучение модели
norm_theta0, norm_theta1 = gradient_descent(norm_mileage, norm_price, learning_rate, epochs)

theta0 = norm_theta0 * np.std(price) + np.mean(price) - norm_theta1 * np.std(price) * np.mean(mileage) / np.std(mileage)
theta1 = norm_theta1 * np.std(price) / np.std(mileage)

# Сохранение параметров
with open('result/theta_values.txt', 'w') as f:
    f.write(f"{theta0},{theta1}")

# Визуализация
plt.scatter(mileage, price, label='Данные')
plt.plot(mileage, theta0 + theta1 * mileage, 'r-', label='Линейная регрессия')
plt.xlabel('Пробег, км')
plt.ylabel('Цена')
plt.legend()
plt.show()