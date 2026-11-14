import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')
sns.set_palette("husl")

# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================
file_path = "C:/Users/User/Desktop/Мои_проекты/Титаник МО/train.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.lower()

print("="*50)
print("ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ")
print("="*50)
print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок\n")

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
df['family_size'] = df['sibsp'] + df['parch'] + 1

def extract_title(name):
    try:
        return str(name).split(',')[1].split('.')[0].strip()
    except IndexError:
        return 'Unknown'

df['title'] = df['name'].apply(extract_title)
rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Capt', 'Sir', 'Lady',
               'Don', 'Jonkheer', 'Mme', 'Ms', 'Mlle']
df['title'] = df['title'].replace(rare_titles, 'Rare')

df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Новые признаки
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                         labels=['ребёнок', 'подросток', 'молодой',
                                 'взрослый', 'пожилой'])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['title', 'age_group'], drop_first=True)

# Собираем признаки
title_cols = [col for col in df.columns if col.startswith('title_')]
age_cols = [col for col in df.columns if col.startswith('age_group_')]
features = ['pclass', 'sex', 'age', 'fare', 'family_size', 'is_alone'] + title_cols + age_cols

X = df[features].copy()
y = df['survived']

# ==========================================
# 3. РАЗДЕЛЕНИЕ, IMPUTER И МАСШТАБИРОВАНИЕ (ИСПРАВЛЕНО!)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Защита от NaN (обучаем imputer ТОЛЬКО на train!)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Масштабирование (обучаем scaler ТОЛЬКО на train!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Переводим обратно в DataFrame для красивых графиков важности признаков
X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)

# ==========================================
# 4. ОБУЧЕНИЕ
# ==========================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

# ==========================================
# 5. ВЫВОД МЕТРИК
# ==========================================
print("="*50)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
print("="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели (Accuracy): {accuracy*100:.2f}%\n")

print("Матрица ошибок (Confusion Matrix):")
cm = confusion_matrix(y_test, y_pred)
print(cm, "\n")

print("Отчёт по классам:")
print(classification_report(y_test, y_pred, target_names=['Погиб', 'Выжил']))

# ==========================================
# 6. ВИЗУАЛИЗАЦИЯ
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# График 1: Важность признаков
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
coefs.plot(kind='barh', ax=axes[0], color='teal')
axes[0].set_title("Важность признаков (коэффициенты)")
axes[0].set_xlabel("Влияние на выживание")

# График 2: Матрица ошибок
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
axes[1].set_title("Матрица ошибок")
axes[1].set_xlabel("Предсказано")
axes[1].set_ylabel("Фактически")
axes[1].set_xticklabels(['Погиб', 'Выжил'])
axes[1].set_yticklabels(['Погиб', 'Выжил'])

# График 3: Распределение вероятностей
probs = model.predict_proba(X_test_final)[:, 1]
axes[2].hist(probs, bins=20, color='coral', edgecolor='black')
axes[2].set_title("Распределение вероятностей выживания")
axes[2].set_xlabel("Вероятность выжить")
axes[2].set_ylabel("Количество пассажиров")

plt.tight_layout()
plt.savefig("titanic_analysis_results.png", dpi=100)
plt.show()

print("\n" + "="*50)
print("Графики сохранены в файл: titanic_analysis_results.png")
print("="*50)
