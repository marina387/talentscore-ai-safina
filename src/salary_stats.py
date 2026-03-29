# Анализ зарплат кандидатов

def mean(values: list[float]) -> float:
    """Средняя зарплата кандидатов"""
    if len(values) == 0:
        raise ValueError("Нет данных о зарплатах")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    """Медианная зарплата (устойчива к выбросам)"""
    if not values:
        raise ValueError("Нет данных о зарплатах")
    sorted_values = sorted(values)
    n = len(sorted_values)
    middle = n // 2
    if n % 2 == 1:
        return float(sorted_values[middle])
    else:
        return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def variance_sample(values: list[float]) -> float:
    """Разброс зарплатных ожиданий"""
    n = len(values)
    if n < 2:
        raise ValueError("Нужно минимум 2 кандидата")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def std_sample(values: list[float]) -> float:
    """Стандартное отклонение зарплат (для поиска аномалий)"""
    return variance_sample(values) ** 0.5


def trimmed_mean(values: list[float], k: int = 1) -> float:
    """Средняя зарплата без учета выбросов"""
    n = len(values)
    if not values:
        raise ValueError("Нет данных")
    if 2 * k >= n:
        raise ValueError("Слишком много выбросов")
    sorted_values = sorted(values)
    core = sorted_values[k:n - k]
    return mean(core)


if __name__ == "__main__":
    # Данные: зарплаты кандидатов (в тыс. руб)
    candidates_salary = [120, 150, 110, 180, 250, 130, 140, 600, 135, 125]

    print("=" * 60)
    print("💰 АНАЛИЗ ЗАРПЛАТ КАНДИДАТОВ (тыс. руб)")
    print("=" * 60)
    print(f"Количество кандидатов: {len(candidates_salary)}")
    print("-" * 40)
    print(f"Средняя зарплата: {mean(candidates_salary):.1f}")
    print(f"Медианная зарплата: {median(candidates_salary):.1f}")
    print(f"Дисперсия: {variance_sample(candidates_salary):.1f}")
    print(f"Стандартное отклонение: {std_sample(candidates_salary):.1f}")
    print(f"Среднее без выбросов (k=1): {trimmed_mean(candidates_salary, k=1):.1f}")

    # Поиск аномалий
    avg = mean(candidates_salary)
    std = std_sample(candidates_salary)
    print("\n🔍 Аномально высокие зарплаты (выше среднего + 2σ):")
    for salary in candidates_salary:
        if salary > avg + 2 * std:
            print(f"  - {salary} (выше среднего на {salary - avg:.0f})")
    print("=" * 60)