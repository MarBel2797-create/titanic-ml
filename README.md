# Titanic Survival Prediction: Сравнение моделей машинного обучения

Проект по анализу данных и построению моделей машинного обучения для предсказания выживаемости пассажиров «Титаника».

## 🎯 Задачи проекта
1. Провести разведочный анализ данных (EDA).
2. Выполнить Feature Engineering (создание признаков "Титул", "Размер семьи", "Возрастные группы").
3. Обучить и сравнить три модели классификации: Random Forest, Logistic Regression и Decision Tree.

## 🛠️ Инструменты
- **Python** (pandas, numpy, matplotlib, seaborn)
- **Scikit-learn** (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, StandardScaler, SimpleImputer)

## 📊 Результаты исследования

В ходе работы были обучены и протестированы три алгоритма классификации.

| Модель | Точность (Accuracy) |
| :--- | :--- |
| **Random Forest** (базовый) | 80.40% |
| **Decision Tree** (базовый) | 80.40% |
| **Logistic Regression** (базовый) | 81.00% |
| **Logistic Regression** (с Feature Engineering и StandardScaler) | **81.01%** |

**Вывод:** Лучший результат показала **Логистическая регрессия (81.01%)**. Добавление сложных признаков (титул, возрастные группы) не дало значительного прироста точности по сравнению с базовой моделью, однако применение масштабирования (`StandardScaler`) и защита от утечки данных позволили сделать модель максимально устойчивой и профессиональной.

**Анализ важности признаков:**
Наибольшее влияние на предсказание выживаемости оказали признаки **пол (sex)**, **класс каюты (pclass)** и **цена билета (fare)**.

## 🚀 Как запустить
В репозитории находятся файлы Python с разными моделями. Финальная версия с полным Feature Engineering — `Titanic_ML_final.py`.

1. Установите зависимости:
   `pip install pandas scikit-learn matplotlib seaborn`
2. Запустите финальный скрипт:
   `python titanic_final.py`

## 📂 Источник данных
[Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/competitions/titanic/data)
