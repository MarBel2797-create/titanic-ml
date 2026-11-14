# ПРОЕКТ: АНАЛИЗ ДАННЫХ И ПРОГНОЗ ВЫЖИВАЕМОСТИ НА ТИТАНИКЕ
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ----------------------------
url = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
df = pd.read_csv("train.csv", sep=None, engine="python")

import csv

# Приводим заголовки к нижнему регистру до проверки.
df.columns = df.columns.str.strip().str.lower()

# Пересобираем строки, если целая запись CSV попала в первый столбец.
if (
    "passengerid" in df.columns
    and "survived" in df.columns
    and df["survived"].isna().all()
):
    packed_rows = df["passengerid"].dropna().astype(str)
    rows = [next(csv.reader([line])) for line in packed_rows]

    if not rows or any(len(row) != len(df.columns) for row in rows):
        raise ValueError("Не удалось повторно разобрать строки CSV.")

    df = pd.DataFrame(rows, columns=df.columns)

# Преобразуем нужные столбцы в числа после повторного разбора.
for column in ["survived", "pclass", "age", "sibsp", "parch", "fare"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print(df[["passengerid", "survived", "pclass"]].head())
df.info()

print("Колонки:", df.columns.tolist())
print("Первая строка:", df.iloc[0].tolist())
print("Форма:", df.shape)

print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

# ----------------------------
# 2. РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA)
# ----------------------------
print("\n--- Общая информация ---")
print(df.info())

# Выживаемость в целом
print(f"\nДоля выживших: {df['survived'].mean()*100:.1f}%")

# Визуализация: выживаемость по полу и классу
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Пол (в этом датасете колонка 'sex', значения 'male'/'female')
sns.barplot(x='sex', y='survived', data=df, ax=axes[0])
axes[0].set_title('Выживаемость по полу')
axes[0].set_ylabel('Доля выживших')

# Класс (в этом датасете колонка 'class', значения 'First', 'Second', 'Third')
sns.barplot(x='pclass', y='survived', data=df, ax=axes[1], order=['First', 'Second', 'Third'])
axes[1].set_title('Выживаемость по классу каюты')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# ----------------------------
# 3. ПОДГОТОВКА ДАННЫХ (Feature Engineering)
# ----------------------------
# Заполняем пропуски в возрасте медианой
df['age'] = df['age'].fillna(df['age'].median())

# Преобразуем текстовые категории в числовые (One-Hot Encoding)
# Для простоты используем только базовые признаки
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
X = df[features].copy()
y = df['survived']

# Превращаем пол в 0/1
X['sex'] = X['sex'].map({'male': 0, 'female': 1})

# Разделяем на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. ОБУЧЕНИЕ МОДЕЛИ (Random Forest)
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Точность модели: {accuracy*100:.1f}%")

print("\nОтчёт по классам:")
print(classification_report(y_test, y_pred, target_names=['Погиб', 'Выжил']))

# ----------------------------
# 5. ВАЖНОСТЬ ПРИЗНАКОВ (что влияло сильнее всего)
# ----------------------------
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', figsize=(8, 4), color='skyblue')
plt.title('Важность признаков для выживания')
plt.xlabel('Вклад в предсказание')
plt.savefig('chart1.png')
plt.show()

print("\nГотово! Вы обучили модель Random Forest.")
