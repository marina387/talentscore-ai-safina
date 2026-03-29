# src/math_stats.py
# Доверительные интервалы и бутстрэп

import numpy as np
from typing import List, Tuple


def mean(values: list[float]) -> float:
    """Среднее арифметическое"""
    if len(values) == 0:
        raise ValueError("mean: empty list")
    return sum(values) / len(values)


def std_sample(values: list[float]) -> float:
    """Выборочное стандартное отклонение (деление на n-1)"""
    n = len(values)
    if n < 2:
        raise ValueError("std_sample: need at least 2 values")
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return var ** 0.5


def ci_mean_normal_approx(values: list[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Приближённый доверительный интервал для среднего:
    mean ± z * (std / sqrt(n))

    confidence: уровень доверия (0.95 → z≈1.96)
    """
    if not values:
        raise ValueError("ci_mean_normal_approx: empty list")
    if confidence <= 0 or confidence >= 1:
        raise ValueError("ci_mean_normal_approx: confidence must be in (0,1)")

    n = len(values)
    m = mean(values)
    sem = std_sample(values) / (n ** 0.5)

    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    margin = z * sem
    return (m - margin, m + margin)


def bootstrap_ci_mean(
        values: list[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42
) -> Tuple[float, float]:
    """
    Bootstrap доверительный интервал для среднего

    - n_bootstrap: количество псевдовыборок
    - confidence: уровень доверия (например, 0.95)

    Возвращает (lower_bound, upper_bound)
    """
    if not values:
        raise ValueError("bootstrap_ci_mean: empty list")
    if n_bootstrap < 100:
        raise ValueError("bootstrap_ci_mean: n_bootstrap too small")

    rng = np.random.default_rng(seed)
    n = len(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.quantile(bootstrap_means, alpha)
    upper = np.quantile(bootstrap_means, 1 - alpha)

    return (float(lower), float(upper))


if __name__ == "__main__":
    # Зарплаты кандидатов (тыс. руб)
    salaries = [120, 150, 110, 180, 250, 130, 140, 600, 135, 125]

    print("=" * 60)
    print(" ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ДЛЯ СРЕДНЕЙ ЗАРПЛАТЫ")
    print("=" * 60)
    print(f"Данные: {salaries}")
    print(f"Количество кандидатов: {len(salaries)}")
    print(f"Средняя зарплата: {mean(salaries):.1f} тыс. руб")
    print(f"Стандартное отклонение: {std_sample(salaries):.1f} тыс. руб")
    print("-" * 40)

    ci_lower, ci_upper = ci_mean_normal_approx(salaries, confidence=0.95)
    print(f"95% CI (нормальное приближение): [{ci_lower:.1f}, {ci_upper:.1f}]")

    ci_lower, ci_upper = bootstrap_ci_mean(salaries, n_bootstrap=1000, confidence=0.95)
    print(f"95% CI (бутстрэп): [{ci_lower:.1f}, {ci_upper:.1f}]")

    print("\n Интерпретация:")
    print("  Мы на 95% уверены, что истинная средняя зарплата")
    print("  по рынку находится в указанном интервале.")
    print("=" * 60)