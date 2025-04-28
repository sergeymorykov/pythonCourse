import pandas as pd

"""Проверка, является ли столбец числовым."""
def is_numeric(column):
  try:
    pd.to_numeric(column)
    return True
  except ValueError:
    return False
    
"""Вычисление статистических метрик для числового столбца."""
def calculate_statistics(column):
  values = column.values
    
  count = len(values)
    
  mean = sum(value / count for value in values)
    
  variance = sum((value - mean) ** 2 / count for value in values)
  std = variance ** 0.5
    
  min_val = min(values)
  max_val = max(values)
    
  sorted_values = sorted(values)
  q25 = sorted_values[int(0.25 * count)]
  q50 = sorted_values[int(0.50 * count)]
  q75 = sorted_values[int(0.75 * count)]
    
  return {
    "count": count,
    "mean": mean,
    "std": std,
    "min": min_val,
    "25%": q25,
    "50%": q50,
    "75%": q75,
    "max": max_val
  }