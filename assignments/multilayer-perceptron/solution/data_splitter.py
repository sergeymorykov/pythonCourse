import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/data.csv')

key_to_value = {
  'diagnosis': '1'
}


df[key_to_value['diagnosis']] = df[key_to_value['diagnosis']].map({'M': 1, 'B': 0})
    
X = df.drop(key_to_value['diagnosis'], axis=1)
y = df[key_to_value['diagnosis']]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
    
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
    
print("Данные успешно разделены.")
