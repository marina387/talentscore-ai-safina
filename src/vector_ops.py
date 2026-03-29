# Операции с векторами для работы с эмбеддингами

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def create_vector(data: list) -> np.ndarray:
    """Создание вектора из списка чисел"""
    return np.array(data)


def vector_length(v: np.ndarray) -> float:
    """Длина (норма) вектора"""
    return float(np.linalg.norm(v))


def vector_sum(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Поэлементное сложение двух векторов"""
    return v1 + v2


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Скалярное произведение двух векторов"""
    return float(np.dot(v1, v2))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Косинусная близость двух векторов

    Возвращает значение от -1 до 1:
    - 1: векторы одинаково направлены (очень похожи)
    - 0: векторы перпендикулярны (не связаны)
    - -1: векторы противоположны
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def plot_vectors(v1: np.ndarray, v2: np.ndarray,
                 xlim: Tuple[int, int] = (0, 4),
                 ylim: Tuple[int, int] = (0, 4),
                 title: str = "Два вектора"):
    """Рисует два вектора как стрелки на плоскости"""
    plt.figure(figsize=(6, 6))

    plt.quiver(0, 0, v1[0], v1[1],
               angles='xy', scale_units='xy', scale=1,
               color='blue', label=f'v1 {v1}')
    plt.quiver(0, 0, v2[0], v2[1],
               angles='xy', scale_units='xy', scale=1,
               color='red', label=f'v2 {v2}')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block=True)  # block=True — окно не закрывается автоматически


if __name__ == "__main__":
    # Создаём два вектора
    v1 = create_vector([2, 1])
    v2 = create_vector([1, 3])

    print("=" * 50)
    print("ОПЕРАЦИИ С ВЕКТОРАМИ")
    print("=" * 50)
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print("-" * 50)
    print(f"Длина v1: {vector_length(v1):.3f}")
    print(f"Длина v2: {vector_length(v2):.3f}")
    print(f"Сумма v1 + v2: {vector_sum(v1, v2)}")
    print(f"Скалярное произведение: {dot_product(v1, v2)}")
    print(f"Косинусная близость: {cosine_similarity(v1, v2):.3f}")
    print("=" * 50)

    # Рисуем график
    plot_vectors(v1, v2)