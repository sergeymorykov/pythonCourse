import pandas as pd
from utils import is_numeric, calculate_statistics

data_train = pd.read_csv('data/dataset_train.csv')
data_test = pd.read_csv('data/dataset_test.csv')
data = pd.concat([data_train, data_test], axis=0)

# Определяем числовые столбцы
numeric_columns = {}
for column_name in data.columns:
  column = data[column_name]
  if is_numeric(column):
    numeric_columns[column_name] = column

# Вычисляем статистику для числовых столбцов
stats = {}
for header, column in numeric_columns.items():
  stats[header] = calculate_statistics(column)

# Выводим статистику
df_stats = pd.DataFrame(stats)
print(df_stats)