# 📦 src/ — исходный код

## 📁 Структура

```text
src/
├── salary_stats.py        # анализ зарплат
├── probability_scoring.py # Байес и скоринг
├── vector_ops.py          # операции с векторами
├── matrix_ops.py          # операции с матрицами
├── math_stats.py          # доверительные интервалы
└── stats_demo.py          # демонстрация
```

## 📄 Описание файлов

| Файл                     | Назначение                     | Основные функции                                                |
| ------------------------ | ------------------------------ | --------------------------------------------------------------- |
| `salary_stats.py`        | Анализ зарплат, поиск аномалий | `mean`, `median`, `std_sample`, `trimmed_mean`                  |
| `probability_scoring.py` | Байесовский скоринг кандидатов | `build_binary_counts`, `bayes_posterior`, `laplace_smooth_prob` |
| `vector_ops.py`          | Операции с векторами           | `dot_product`, `cosine_similarity`                              |
| `matrix_ops.py`          | Операции с матрицами           | `matrix_vector_multiply`, `rank_candidates_by_scores`           |
| `math_stats.py`          | Доверительные интервалы        | `ci_mean_normal_approx`, `bootstrap_ci_mean`                    |
| `stats_demo.py`          | Демонстрация на HR-данных      | —                                                               |

## 🔧 Зависимости

* `numpy`
* `matplotlib`

## 📌 Версия

`v0.4` — модуль 4 завершён (статистика, вероятность, линейная алгебра)
