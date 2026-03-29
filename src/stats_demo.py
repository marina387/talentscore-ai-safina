# src/stats_demo.py
# Демонстрация статистических методов в HR-контексте

import numpy as np
import matplotlib.pyplot as plt


def generate_embeddings(n_candidates: int = 1000, mean: float = 0, std: float = 1) -> np.ndarray:
    """Генерация тестовых эмбеддингов кандидатов"""
    return np.random.normal(mean, std, n_candidates)


def analyze_distribution(embeddings: np.ndarray) -> dict:
    """Анализ распределения эмбеддингов"""
    return {
        "mean": np.mean(embeddings),
        "std": np.std(embeddings),
        "median": np.median(embeddings),
        "n": len(embeddings)
    }


def find_anomalies(embeddings: np.ndarray, stats: dict, sigma: float = 3) -> np.ndarray:
    """Поиск аномальных кандидатов (отклонение > sigma * std)"""
    return embeddings[np.abs(embeddings - stats["mean"]) > sigma * stats["std"]]


def plot_distribution(embeddings: np.ndarray, stats: dict, title: str = "Распределение кандидатов"):
    """Построение гистограммы распределения"""
    plt.figure(figsize=(10, 5))
    plt.hist(embeddings, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(stats["mean"], color='red', linestyle='--', linewidth=2, label=f"Среднее ({stats['mean']:.3f})")
    plt.axvline(stats["median"], color='green', linestyle='--', linewidth=2, label=f"Медиана ({stats['median']:.3f})")
    plt.title(title)
    plt.xlabel("Эмбеддинг")
    plt.ylabel("Количество кандидатов")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print(" ДЕМОНСТРАЦИЯ СТАТИСТИКИ ДЛЯ HR-СИСТЕМЫ")
    print("=" * 60)

    # Генерация тестовых эмбеддингов
    n_candidates = 1000
    embeddings = generate_embeddings(n_candidates)
    print(f"✅ Сгенерировано {len(embeddings)} тестовых эмбеддингов")

    # Анализ распределения
    stats = analyze_distribution(embeddings)
    print("\n Статистика по кандидатам:")
    print(f"  Среднее: {stats['mean']:.4f}")
    print(f"  Медиана: {stats['median']:.4f}")
    print(f"  Стандартное отклонение: {stats['std']:.4f}")

    # Правило 68-95-99.7
    within_1std = np.sum(np.abs(embeddings - stats["mean"]) <= stats["std"])
    print(f"\n Кандидаты в пределах 1σ: {within_1std / n_candidates * 100:.1f}% (теория: ~68%)")

    # Поиск аномалий
    anomalies = find_anomalies(embeddings, stats, sigma=3)
    print(f"\n Аномальных кандидатов (>3σ): {len(anomalies)}")

    # Визуализация
    print("\n Построение графика...")
    plot_distribution(embeddings, stats)

    print("\n✅ Демонстрация завершена")
    print("=" * 60)