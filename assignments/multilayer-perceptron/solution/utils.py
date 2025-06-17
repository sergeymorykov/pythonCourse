import numpy as np
import pandas as pd
import pickle

# Активационные функции
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Нормализация
def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

# Прямое распространение
def forward_pass(X, weights):
    activations = [X]
    for W in weights[:-1]:
        Z = np.dot(activations[-1], W)
        A = sigmoid(Z)  # Sigmoid для скрытых слоёв
        activations.append(A)
    # Выходной слой — softmax
    Z_out = np.dot(activations[-1], weights[-1])
    A_out = softmax(Z_out)
    activations.append(A_out)
    return activations

# Загрузка весов
def load_weights(path='result/weights.npy'):
    with open(path, 'rb') as f:
        weights = pickle.load(f)
    return weights

# One-hot кодирование
def to_one_hot(y, num_classes=2):
    y = np.asarray(y, dtype=np.int64).flatten()
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

# Функции потерь и метрик
def categorical_cross_entropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def accuracy(y_pred, y_true):
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    return np.mean(y_pred_labels == y_true_labels)

# Загрузка данных
def load_data(path='data/train.csv'):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    return X, y 