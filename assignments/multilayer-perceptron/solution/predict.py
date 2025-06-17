import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import (
    standardize, forward_pass,
    load_weights, to_one_hot, categorical_cross_entropy,
    accuracy, load_data
)

# Предсказание одного образца
def predict_sample(sample, weights):
    sample = np.array([sample])  # преобразуем в batch_size=1
    output = forward_pass(sample, weights)[-1]  # получаем выходной слой
    return 'M' if np.argmax(output[0]) == 1 else 'B'

# Оценка модели
def evaluate(X, y, weights):
    predictions = forward_pass(X, weights)[-1]
    loss = categorical_cross_entropy(predictions, y)
    acc = accuracy(predictions, y)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    plot_results(y, predictions)

# Визуализация результатов
def plot_results(y_true, y_pred_prob):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_prob, axis=1)

    assert len(y_true_labels) == len(y_pred_labels), "Размерности меток не совпадают"

    # Создаем папку для графиков
    os.makedirs('result/metrics', exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['B', 'M'], yticklabels=['B', 'M'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('result/metrics/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    weights = load_weights()

    X_new, Y_new = load_data('data/test.csv')
    X_new = standardize(X_new)
    Y_onehot = to_one_hot(Y_new, num_classes=2)

    evaluate(X_new, Y_onehot, weights)