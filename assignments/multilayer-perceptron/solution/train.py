import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import os
from utils import (
    standardize, forward_pass,
    to_one_hot, categorical_cross_entropy,
    accuracy, load_data
)

def sigmoid_derivative(x):
    return x * (1 - x)

# Инициализация весов
def init_weights(sizes):
    weights = []
    for i in range(len(sizes) - 1):
        weights.append(np.random.randn(sizes[i], sizes[i+1]) * 0.1)
    return weights

# Обратное распространение
def backward_pass(activations, y, weights, learning_rate):
    deltas = [None] * len(weights)
    deltas[-1] = activations[-1] - y

    # Распространение ошибки назад
    for l in range(len(weights)-2, -1, -1):
        error = np.dot(deltas[l+1], weights[l+1].T)
        deltas[l] = error * sigmoid_derivative(activations[l+1])

    # Обновление весов
    for l in range(len(weights)):
        weights[l] -= learning_rate * np.dot(activations[l].T, deltas[l])

# Сохранение весов
def save_weights(weights, path='result/weights.npy'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(weights, f)

# Графики метрик
def plot_metrics(losses, accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses['train'], label='training loss')
    plt.plot(losses['val'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies['train'], label='training acc')
    plt.plot(accuracies['val'], label='validation acc')
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Цикл обучения
def train(X_train, y_train, hidden_sizes=[24, 24, 24], epochs=84, batch_size=8, lr=0.0314, data_valid=None):
    input_size = X_train.shape[1]
    output_size = 2  # Теперь 2 нейрона на выходе
    sizes = [input_size] + hidden_sizes + [output_size]
    weights = init_weights(sizes)

    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = idx[i:i+batch_size]
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

            acts = forward_pass(X_batch, weights)
            backward_pass(acts, y_batch, weights, lr)

        # Метрики на тренировочной выборке
        pred_train = forward_pass(X_train, weights)[-1]
        loss_train = categorical_cross_entropy(pred_train, y_train)
        acc_train = accuracy(pred_train, y_train)
        losses['train'].append(loss_train)
        accuracies['train'].append(acc_train)

        # Валидация
        if data_valid:
            X_val, y_val = data_valid
            pred_val = forward_pass(X_val, weights)[-1]
            loss_val = categorical_cross_entropy(pred_val, y_val)
            acc_val = accuracy(pred_val, y_val)
            losses['val'].append(loss_val)
            accuracies['val'].append(acc_val)
            print(f"epoch {epoch+1}/{epochs} - loss: {loss_train:.4f} - val loss: {loss_val:.4f} - val acc: {acc_val:.4f}")
        else:
            print(f"epoch {epoch+1}/{epochs} - loss: {loss_train:.4f}")

    save_weights(weights)
    plot_metrics(losses, accuracies)
    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение многослойного перцептрона')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=8, help='Размер батча')
    parser.add_argument('--learning-rate', type=float, default=0.005, help='Скорость обучения')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[24, 24, 24], help='Размеры скрытых слоев')
    
    args = parser.parse_args()
    
    X_train, y_train = load_data('data/train.csv')
    X_test, y_test = load_data('data/test.csv')

    X_train = standardize(X_train)
    X_test = standardize(X_test)

    # Переводим метки в one-hot
    y_train_onehot = to_one_hot(y_train, num_classes=2)
    y_test_onehot = to_one_hot(y_test, num_classes=2)

    train(X_train, y_train_onehot,
          hidden_sizes=args.hidden_sizes,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.learning_rate,
          data_valid=(X_test, y_test_onehot))