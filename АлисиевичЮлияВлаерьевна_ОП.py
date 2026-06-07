import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Установка seed для воспроизводимости результатов
np.random.seed(42)

# 1. Генерация численных данных
print("=" * 60)
print("ГЕНЕРАЦИЯ ДАННЫХ")
print("=" * 60)

# Генерируем 1000 целых чисел в диапазоне от -10000 до 10000
data = np.random.randint(-10000, 10001, 1000)

# Создаем объект Series
series = pd.Series(data, name='Исходные_данные')

print(f"Сгенерировано {len(series)} чисел")
print(f"Первые 10 значений: {series.head(10).tolist()}")
print()

# 2. Расчет стандартных числовых характеристик
print("=" * 60)
print("СТАНДАРТНЫЕ ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ")
print("=" * 60)

# Определение минимального значения
min_value = series.min()
print(f"Минимальное значение: {min_value}")

# Определение количества повторяющихся значений
# (уникальные значения и их частоты)
value_counts = series.value_counts()
duplicates_count = (value_counts > 1).sum()  # количество значений, которые встречаются >1 раза
total_duplicates = (len(series) - len(series.unique()))  # общее количество повторяющихся элементов
print(f"Количество уникальных значений: {len(series.unique())}")
print(f"Количество повторяющихся значений (элементов): {total_duplicates}")
print(f"Количество значений, имеющих дубликаты: {duplicates_count}")

# Определение максимального значения
max_value = series.max()
print(f"Максимальное значение: {max_value}")

# Определение суммы чисел
sum_value = series.sum()
print(f"Сумма всех чисел: {sum_value}")

# Определение среднеквадратического отклонения
std_dev = series.std()
print(f"Среднеквадратическое отклонение: {std_dev:.4f}")

# Дополнительные статистики для полноты картины
print(f"\nДополнительные статистики:")
print(f"Среднее арифметическое: {series.mean():.4f}")
print(f"Медиана: {series.median()}")
print(f"Дисперсия: {series.var():.4f}")

# 3. Визуализация исходных данных
print("\n" + "=" * 60)
print("ВИЗУАЛИЗАЦИЯ ДАННЫХ")
print("=" * 60)

# Создаем фигуру с двумя подграфиками
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Линейный график (первые 100 точек для наглядности)
axes[0].plot(series.index[:100], series.values[:100],
             color='blue', marker='o', markersize=3, linewidth=1)
axes[0].set_title('Линейный график исходных данных (первые 100 точек)', fontsize=14)
axes[0].set_xlabel('Индекс')
axes[0].set_ylabel('Значение')
axes[0].grid(True, alpha=0.3)

# Гистограмма с округлением до сотен
# Округляем значения по математическому правилу
rounded_data = np.round(series.values / 100) * 100
axes[1].hist(rounded_data, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1].set_title('Гистограмма данных (округленных до сотен)', fontsize=14)
axes[1].set_xlabel('Значение (округленное до сотен)')
axes[1].set_ylabel('Частота')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Формирование DataFrame с отсортированными данными
print("\n" + "=" * 60)
print("ФОРМИРОВАНИЕ DATAFRAME")
print("=" * 60)

# Создаем DataFrame из исходного Series
df = pd.DataFrame({
    'Исходные_данные': series
})

# Добавляем столбец с сортировкой по возрастанию
df['Сортировка_по_возрастанию'] = series.sort_values().reset_index(drop=True)

# Добавляем столбец с сортировкой по убыванию
df['Сортировка_по_убыванию'] = series.sort_values(ascending=False).reset_index(drop=True)

print(f"DataFrame создан. Размер: {df.shape}")
print("\nПервые 10 строк DataFrame:")
print(df.head(10))
print("\nПоследние 10 строк DataFrame:")
print(df.tail(10))

# 5. Визуализация отсортированных данных
print("\n" + "=" * 60)
print("ВИЗУАЛИЗАЦИЯ ОТСОРТИРОВАННЫХ ДАННЫХ")
print("=" * 60)

# Создаем график с двумя линейными графиками
plt.figure(figsize=(12, 6))

# График отсортированных значений по возрастанию
plt.plot(df.index, df['Сортировка_по_возрастанию'],
         color='red', linewidth=2, label='По возрастанию', alpha=0.8)

# График отсортированных значений по убыванию
plt.plot(df.index, df['Сортировка_по_убыванию'],
         color='blue', linewidth=2, label='По убыванию', alpha=0.8)
plt.title('Сравнение отсортированных значений (возрастание vs убывание)', fontsize=16)
plt.xlabel('Индекс в отсортированном порядке', fontsize=12)
plt.ylabel('Значение', fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Дополнительная статистика по отсортированным даннымф
print("\n" + "=" * 60)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
print("=" * 60)

print(f"Проверка: Сортировка по возрастанию - первый элемент = {df['Сортировка_по_возрастанию'].iloc[0]}")
print(f"Проверка: Сортировка по возрастанию - последний элемент = {df['Сортировка_по_возрастанию'].iloc[-1]}")
print(f"Проверка: Сортировка по убыванию - первый элемент = {df['Сортировка_по_убыванию'].iloc[0]}")
print(f"Проверка: Сортировка по убыванию - последний элемент = {df['Сортировка_по_убыванию'].iloc[-1]}")

# Проверка, что отсортированные данные являются обратными друг другу
is_reverse = (df['Сортировка_по_возрастанию'].values == df['Сортировка_по_убыванию'].values[::-1]).all()
print(f"Являются ли отсортированные ряды обратными друг другу: {is_reverse}")

# Информация о DataFrame
print("\nИнформация о DataFrame:")
print(df.info())

# Статистика DataFrame
print("\nСтатистическое описание DataFrame:")
print(df.describe())