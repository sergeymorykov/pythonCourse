import numpy as np

# Загрузка обученных параметров
with open('result/theta_values.txt', 'r') as f:
    theta0, theta1 = map(float, f.read().split(','))

mileage = float(input("Введите пробег автомобиля: "))

# Предсказание цены
price = theta0 + theta1 * mileage
print(f"Предполагаемая цена: {price:.2f}")