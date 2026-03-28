# Лабораторная работа №4. Анализ и обработка больших данных с Dask (ETL-пайплайн)

**Вариант 14**: `Parking_Violations_Issued_-_Fiscal_Year_2015.csv` (2.8 ГБ)

**Цель работы:** изучить инструменты Dask для обработки Big Data, освоить построение ETL-пайплайнов с «ленивыми вычислениями» и визуализировать графы выполнения задач (DAG).

## Шаг 1. Extract (Извлечение данных)

Для работы с файлом объемом 2.8 ГБ (около 11 млн строк) используется `dask.dataframe`. Настраиваем локальный кластер для параллельной обработки.

```python
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar

# Инициализация клиента Dask (2 воркера, 2 потока на воркер)
client = Client(n_workers=2, threads_per_worker=2, processes=True)

# Чтение данных с указанием типов для оптимизации
dtypes = {
    'Issuer Command': 'object', 'Issuer Squad': 'object',
    'House Number': 'object', 'Time First Observed': 'object',
    'Violation Description': 'object', 'Violation Legal Code': 'object',
    'Violation Post Code': 'object', 'Unregistered Vehicle?': 'float64',
    'Violation Location': 'float64', 'Date First Observed': 'object',
    'Feet From Curb': 'float64', 'Law Section': 'object',
    'Vehicle Year': 'float64', 'Meter Number': 'object',
    'Violation County': 'object',
    'Double Parking Violation': 'object',
    'Hydrant Violation': 'object',
    'No Standing or Stopping Violation': 'object'
}

df = dd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2015.csv', dtype=dtypes, low_memory=False)
```
Результат:
<img width="1280" height="841" alt="image" src="https://github.com/user-attachments/assets/b19920c8-ab48-4903-8e8f-46c3dcc47d46" />
<img width="1292" height="836" alt="image" src="https://github.com/user-attachments/assets/6c58a947-c6fc-4c2d-a799-89ec92783893" />
<img width="1334" height="596" alt="image" src="https://github.com/user-attachments/assets/9534fba8-2269-4289-a0f0-78178ce97799" />

## Шаг 2. Transform (Трансформация и очистка данных)

Проведено профилирование качества данных (подсчет пропусков). Столбцы с пропуском более 55% удаляются. Также удаляются технические столбцы, не несущие смысловой нагрузки для анализа.

```python
#from dask.diagnostics import ProgressBar

# Подсчет пропущенных значений (построение графа вычислений)
missing_values = df.isnull().sum()

# Вычисление процента пропусков
mysize = df.index.size
missing_count = ((missing_values / mysize) * 100)

# Запуск реальных вычислений только для агрегированной статистики
with ProgressBar():
    missing_count_percent = missing_count.compute()

print(missing_count_percent.sort_values(ascending=False).head(15))

# Формирование списка столбцов, где пропусков > 55%
columns_to_drop = list(missing_count_percent[missing_count_percent > 55].index)
print("\nУдаляемые столбцы (пропуски > 55%):", columns_to_drop)

# Ленивое удаление столбцов
df_dropped = df.drop(columns=columns_to_drop)

# Удаление дополнительных технических и избыточных столбцов
additional_columns = [
    'Street Code1', 'Street Code2', 'Street Code3',
    'Issuer Code', 'Feet From Curb', 'Violation Post Code'
]

existing_extra = [c for c in additional_columns if c in df_dropped.columns]
df_final = df_dropped.drop(columns=existing_extra)

# Преобразование формата даты
df_final['Issue Date'] = dd.to_datetime(df_final['Issue Date'], errors='coerce')

df_final.head()
```

**Результат очистки:**
Удалены столбцы с максимальным количеством пропусков: `NTA`, `BBL`, `BIN`, `Latitude`, `Longitude` и др.
<img width="1680" height="838" alt="image" src="https://github.com/user-attachments/assets/e3f8edad-a51a-4d63-a10c-8d38005fd7e6" />

## Шаг 3. Load (Загрузка / Сохранение результатов)

Очищенный датасет сохраняется в формате Parquet, который является стандартом де-факто для больших данных благодаря колоночному хранению и высокой скорости чтения/записи в Dask.

```python
df_final.to_parquet('cleaned_violations_2015.parquet', engine='pyarrow')
```
Результат:
<img width="1754" height="275" alt="image" src="https://github.com/user-attachments/assets/621361f8-5e11-4e56-aafa-b2fc458d65d1" />

## Визуализация DAG

### 1. Простой граф (Аналитика марок ТС)

Граф визуализирует процесс подсчета общего числа строк, уникальных марок автомобилей и вычисление среднего значения нарушений на марку.

```python
from dask import delayed
from IPython.display import Image

def get_total_violations():
    return len(df_final)

def get_unique_makes():
    return df_final['Vehicle Make'].nunique().compute()

def avg_violations_per_make(total, unique):
    if unique == 0: return 0
    return round(total / unique, 2)

x = delayed(get_total_violations)()
y = delayed(get_unique_makes)()
z = delayed(avg_violations_per_make)(x, y)

try:
    z.visualize(filename='simple_violation_analysis.png')
    display(Image('simple_violation_analysis.png'))
except:
    print("Graphviz not found.")

print("Результат вычисления DAG:", z.compute())
```
Результат: 

<img width="350" height="275" alt="image" src="img/simple_violation_analysis.png" />

### 2. Сложный граф (Анализ по районам NYC)

Построение многоуровневого графа для анализа доли нарушений в часы пик (8:00 - 10:00) в разрезе округов (Violation County).

```python
from dask import delayed
from IPython.display import Image

# Список районов для анализа
districts = ['NY', 'K', 'Q', 'BX', 'R']

def load_district_data(district):
    return df_final[df_final['Violation County'] == district]

def count_violations(district_data):
    return len(district_data)

def count_peak_hours(district_data):
    if district_data is None or len(district_data) == 0: return 0
    # Выделяем часы из Violation Time (первые 2 символа)
    hours = district_data['Violation Time'].astype(str).str[:2]
    peak = hours[hours.isin(['08', '09', '10'])]
    return len(peak)

def calculate_peak_percentage(total, peak):
    if total == 0: return 0
    return round((peak / total) * 100, 2)

layer1 = [delayed(load_district_data)(d) for d in districts]
layer2 = [delayed(count_violations)(d) for d in layer1]
layer3 = [delayed(count_peak_hours)(d) for d in layer1]
layer4 = [delayed(calculate_peak_percentage)(t, p) for t, p in zip(layer2, layer3)]

results = delayed(list)(layer4)

try:
    results.visualize(filename='complex_district_analysis.png')
    display(Image('complex_district_analysis.png'))
except:
    print("Graphviz error.")

print("Результаты по районам % (NY, K, Q, BX, R):", results.compute())
```
Результат:

<img width="350" height="275" alt="image" src="img/complex_district_analysis.png" />

## Аналитика
<img width="1011" height="615" alt="image" src="https://github.com/user-attachments/assets/f6dcfc8b-9225-403a-bb77-75c2d8300996" />
<img width="954" height="613" alt="image" src="https://github.com/user-attachments/assets/595dd4bf-3756-4717-95cc-97369e7a94c1" />
<img width="894" height="604" alt="image" src="https://github.com/user-attachments/assets/62396996-78ae-4b71-a54e-6dea9403fa7a" />
<img width="879" height="624" alt="image" src="https://github.com/user-attachments/assets/d97b5df2-4d47-44a5-9439-7f9e37160e93" />
<img width="1174" height="634" alt="image" src="https://github.com/user-attachments/assets/0e2618de-1cea-40ca-8fe4-f6fc9160c46c" />
<img width="878" height="611" alt="image" src="https://github.com/user-attachments/assets/f24b0195-9737-4c4c-b060-b80eb55313ad" />
<img width="860" height="710" alt="image" src="https://github.com/user-attachments/assets/0e486291-934a-485e-9fa2-f062056f822d" />






