import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Убедитесь, что файл CSV существует
file_path = 'matrix_multiplication_results.csv'
if not os.path.exists(file_path):
    print(f"Ошибка: Файл '{file_path}' не найден. Пожалуйста, сначала запустите C++ CUDA код.")
    exit()

# Загрузка данных из CSV-файла
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Ошибка при чтении CSV-файла: {e}")
    exit()

print("Данные успешно загружены:")
print(df)

# Настройка стиля графиков
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter'] # Предполагается, что Inter доступен

# --- График сравнения времени выполнения ---
plt.figure(figsize=(12, 7))
plt.plot(df['MatrixSize'], df['ClassicalTimeMs'], marker='o', label='Классический алгоритм (CUDA)', color='skyblue', linewidth=2)
plt.plot(df['MatrixSize'], df['StrassenTimeMs'], marker='x', label='Алгоритм Штрассена (гибридный CUDA)', color='salmon', linewidth=2)

plt.title('Сравнение времени выполнения умножения матриц', fontsize=16, fontweight='bold')
plt.xlabel('Размер матрицы (N x N)', fontsize=12)
plt.ylabel('Временная эффективность (1/t)', fontsize=12)
plt.xscale('log', base=2) # Логарифмическая шкала по X, так как размеры матриц - степени двойки
plt.yscale('log') # Логарифмическая шкала по Y для лучшей видимости различий
plt.xticks(df['MatrixSize'], labels=[str(s) for s in df['MatrixSize']]) # Отображение всех меток на оси X
plt.legend(fontsize=10)
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig('time_comparison.png', dpi=300)
plt.show()

# --- График сравнения объемов памяти ---
plt.figure(figsize=(12, 7))
plt.plot(df['MatrixSize'], df['ClassicalMemoryBytes'] / (1024 * 1024), marker='o', label='Классический алгоритм', color='lightgreen', linewidth=2)
plt.plot(df['MatrixSize'], df['StrassenMemoryBytes'] / (1024 * 1024), marker='x', label='Алгоритм Штрассена', color='orange', linewidth=2)

plt.title('Сравнение объемов памяти умножения матриц', fontsize=16, fontweight='bold')
plt.xlabel('Размер матрицы (N x N)', fontsize=12)
plt.ylabel('Объем памяти (МБ)', fontsize=12)
plt.xscale('log', base=2) # Логарифмическая шкала по X
plt.yscale('log') # Логарифмическая шкала по Y
plt.xticks(df['MatrixSize'], labels=[str(s) for s in df['MatrixSize']])
plt.legend(fontsize=10)
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig('memory_comparison.png', dpi=300)
plt.show()

print("Графики сохранены как time_comparison.png и memory_comparison.png")
