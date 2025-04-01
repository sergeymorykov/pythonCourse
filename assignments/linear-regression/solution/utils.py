import numpy as np

"""Реализует алгоритм градиентного спуска для линейной регрессии.

Args:
    X (np.ndarray): Массив признаков.
    Y (np.ndarray): Массив целевых значений.
    learn_h_rate (float): шаг градиентного спуска.
    number_of_iterations (int): Количество итераций градиентного спуска.

Returns:
    tuple: Пара (theta0, theta1), где:
      - theta0 (float): Свободный член уравнения регрессии
      - theta1 (float): Коэффициент наклона уравнения регрессии
"""
def gradient_descent(X, Y, learn_h_rate, number_of_iterations):
    theta0, theta1 = 0, 0
    m = len(X)
    
    for _ in range(number_of_iterations):
        predictions = theta0 + theta1 * X
        tmp_theta0 = learn_h_rate * (1/m) * np.sum(predictions - Y)
        tmp_theta1 = learn_h_rate * (1/m) * np.sum((predictions - Y) * X)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
    
    return theta0, theta1


"""Вычисляет среднеквадратичную ошибку для линейной регрессии.

Args:
    X (np.ndarray): Массив признаков.
    Y (np.ndarray): Массив целевых значений.
    theta0 (float): Свободный член уравнения регрессии.
    theta1 (float): Коэффициент наклона уравнения регрессии.

Returns:
    float: Значение среднеквадратичной ошибки (MSE).
"""
def mean_squared_error(X, Y, theta0, theta1):
    predictions = theta0 + theta1 * X
    return np.mean((predictions - Y) ** 2)