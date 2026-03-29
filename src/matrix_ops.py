# src/matrix_ops.py
# Операции с матрицами для ранжирования кандидатов

import numpy as np
from typing import List, Tuple


def create_candidate_matrix(candidates_data: List[List[float]]) -> np.ndarray:
    """
    Создание матрицы кандидатов из списка списков

    Каждая строка - кандидат, каждый столбец - признак
    (опыт, навыки, зарплата и т.д.)
    """
    return np.array(candidates_data)


def get_matrix_shape(X: np.ndarray) -> Tuple[int, int]:
    """Возвращает размер матрицы (строки, столбцы)"""
    return X.shape


def matrix_vector_multiply(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Умножение матрицы кандидатов на вектор весов

    Для каждого кандидата: сумма(признаки * веса)
    Результат - оценка (рейтинг) каждого кандидата
    """
    if X.shape[1] != len(w):
        raise ValueError(
            f"Число признаков ({X.shape[1]}) должно равняться "
            f"числу весов ({len(w)})"
        )
    return X @ w


def rank_candidates_by_scores(scores: np.ndarray, ascending: bool = False) -> np.ndarray:
    """
    Ранжирование кандидатов по оценкам

    ascending=False - от лучшего к худшему (по убыванию)
    ascending=True - от худшего к лучшему (по возрастанию)

    Возвращает индексы кандидатов в отсортированном порядке
    """
    if ascending:
        return np.argsort(scores)
    else:
        return np.argsort(scores)[::-1]


def get_top_candidates(
        X: np.ndarray,
        scores: np.ndarray,
        top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает топ-K кандидатов и их оценки
    """
    indices = rank_candidates_by_scores(scores)
    top_indices = indices[:top_k]
    return X[top_indices], scores[top_indices]


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Нормализация признаков (каждый столбец приводится к среднему 0, std 1)

    Важно для корректного сравнения признаков с разными масштабами
    """
    X_norm = X.copy().astype(float)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        mean_val = np.mean(col_data)
        std_val = np.std(col_data)
        if std_val > 0:
            X_norm[:, col] = (col_data - mean_val) / std_val
    return X_norm


if __name__ == "__main__":
    # Данные кандидатов: [опыт (годы), знание Python (0/1), знание SQL (0/1)]
    candidates_data = [
        [3, 1, 1],  # кандидат 1
        [5, 1, 0],  # кандидат 2
        [1, 0, 1],  # кандидат 3
        [4, 1, 1],  # кандидат 4
        [2, 0, 0],  # кандидат 5
    ]

    # Веса признаков: [важность опыта, важность Python, важность SQL]
    weights = np.array([10, 5, 3])

    print("=" * 60)
    print("РАНЖИРОВАНИЕ КАНДИДАТОВ")
    print("=" * 60)

    # Создаём матрицу кандидатов
    X = create_candidate_matrix(candidates_data)
    print(f"Матрица кандидатов ({get_matrix_shape(X)[0]} строк, {get_matrix_shape(X)[1]} столбцов):")
    print(X)
    print(f"\nВектор весов: {weights}")

    # Умножаем матрицу на вектор весов
    scores = matrix_vector_multiply(X, weights)
    print(f"\nОценки кандидатов: {scores}")

    # Ранжируем
    ranking = rank_candidates_by_scores(scores)
    print(f"\nРанжирование (индексы от лучшего к худшему): {ranking}")

    print("\n" + "-" * 60)
    print("ТОП-3 КАНДИДАТА:")
    print("-" * 60)

    top_candidates, top_scores = get_top_candidates(X, scores, top_k=3)
    for i, (candidate, score) in enumerate(zip(top_candidates, top_scores)):
        print(f"{i + 1}. Кандидат {candidate} -> оценка: {score:.0f}")

    # Нормализация признаков
    print("\n" + "-" * 60)
    print("НОРМАЛИЗАЦИЯ ПРИЗНАКОВ:")
    print("-" * 60)

    X_normalized = normalize_features(X)
    print("Исходная матрица:")
    print(X)
    print("\nНормализованная матрица (mean=0, std=1 для каждого столбца):")
    print(X_normalized)

    print("\n" + "=" * 60)