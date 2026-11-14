# Titanic Survival Prediction: Сравнение моделей машинного обучения

Проект по анализу данных и построению моделей машинного обучения для предсказания выживаемости пассажиров «Титаника».

## 🎯 Задачи проекта
1. Провести разведочный анализ данных (EDA).
2. Выявить ключевые факторы, влияющие на выживаемость.
3. Обучить и сравнить три модели классификации: Random Forest, Logistic Regression и Decision Tree.

## 🛠️ Инструменты
- **Python** (pandas, numpy, matplotlib, seaborn)
- **Scikit-learn** (RandomForestClassifier, LogisticRegression, DecisionTreeClassifier)

## 📊 Результаты исследования

В ходе работы были обучены и протестированы три алгоритма классификации. Результаты точности (Accuracy) на тестовой выборке:

| Модель | Точность (Accuracy) |
| :--- | :--- |
| **Random Forest** | 80.4% |
| **Logistic Regression** | **81.0%** |
| **Decision Tree** | 80.4% |

**Вывод:** Лучший результат показала **Логистическая регрессия (81%)**. Это говорит о том, что для данного набора данных линейная модель справляется чуть лучше, чем ансамбль деревьев и одиночное дерево решений.

**Анализ важности признаков:**
Разные модели выделили разные главные признаки. Random Forest и Decision Tree склонялись к цене билета (fare) и классу, а Logistic Regression и Decision Tree (в другом запуске) выделили пол (sex) как самый значимый фактор. Это подтверждает исторический факт: при спасении в первую очередь приоритет отдавался женщинам.

## 🚀 Как запустить
В репозитории находятся три файла Python:
- `Titanic_ML1_RandomForest.py`
- `Titanic_ML2_LogisticRegression.py`
- `Titanic_ML3_DecisionTreeClassifier.py`

1. Установите зависимости:
   `pip install pandas scikit-learn matplotlib seaborn`
2. Запустите любой из файлов:
   `Titanic_ML2_LogisticRegression.py`

## 📂 Источник данных
[Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/competitions/titanic/data)
---
