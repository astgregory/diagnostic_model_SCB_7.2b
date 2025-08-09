import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('diagnosis_dataset_7_2b.csv')
X = df['Input']
y = df['Output']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

vectorizer = TfidfVectorizer(stop_words=None)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = SVC(kernel='linear', random_state=42, C=250)
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

plt.ylabel('Истинные метки')
plt.xlabel('Предсказанные метки')
plt.title('Матрица ошибок')
plt.savefig('plots/confusion_matrix.png')

plt.show()

print("Описание классов:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{index} - {label}")

precision, recall, f1_score, support = [], [], [], []
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

for key in report.keys():
    if key not in ['accuracy', 'macro avg', 'weighted avg']:
        precision.append(report[key]['precision'])
        recall.append(report[key]['recall'])
        f1_score.append(report[key]['f1-score'])

classes = label_encoder.classes_

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(12, 7))
plt.bar(x - width, precision, width, label='Точность', color='blue')
plt.bar(x, recall, width, label='Полнота', color='orange')
plt.bar(x + width, f1_score, width, label='F1-Score', color='green')
plt.legend()

joblib.dump(model, 'models/model.joblib')
joblib.dump(vectorizer, 'models/vectorizer.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')

print("Модель и компоненты успешно сохранены.")