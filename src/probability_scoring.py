# Функции для вероятностного скоринга кандидатов

def build_binary_counts(recs: list[dict], a_key: str, b_key: str) -> dict:
    """Подсчёт частот для двух бинарных признаков"""
    n = len(recs)
    count_A = 0
    count_B = 0
    count_A_and_B = 0

    for r in recs:
        a = int(r[a_key])
        b = int(r[b_key])
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("build_binary_counts: values must be 0/1")
        if a == 1:
            count_A += 1
        if b == 1:
            count_B += 1
        if a == 1 and b == 1:
            count_A_and_B += 1

    return {"n": n, "count_A": count_A, "count_B": count_B, "count_A_and_B": count_A_and_B}


def prob_from_counts(count: int, n: int) -> float:
    """Вероятность из частоты: P = count / n"""
    if n <= 0:
        raise ValueError("prob_from_counts: n must be > 0")
    if count < 0 or count > n:
        raise ValueError("prob_from_counts: invalid count")
    return count / n


def prob_conditional(count_A_and_B: int, count_A: int) -> float:
    """Условная вероятность P(B|A) = count(A∩B) / count(A)"""
    if count_A <= 0:
        raise ValueError("prob_conditional: condition count must be > 0")
    if count_A_and_B < 0 or count_A_and_B > count_A:
        raise ValueError("prob_conditional: invalid intersection count")
    return count_A_and_B / count_A


def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    """
    Формула Байеса: P(A|B) = P(B|A) * P(A) / P(B)

    prior = P(A)
    likelihood = P(B|A)
    evidence = P(B)
    """
    for name, p in [("prior", prior), ("likelihood", likelihood), ("evidence", evidence)]:
        if p < 0 or p > 1:
            raise ValueError(f"bayes_posterior: {name} must be in [0,1]")
    if evidence == 0:
        raise ValueError("bayes_posterior: evidence must be > 0")
    return (likelihood * prior) / evidence


def score_by_skill(recs: list[dict], skill_value: int) -> float:
    """
    Вероятность успеха для кандидатов с/без навыка

    skill_value: 1 - есть навык, 0 - нет навыка
    """
    if skill_value not in (0, 1):
        raise ValueError("skill_value must be 0/1")
    subset = [r for r in recs if int(r["python"]) == skill_value]
    if len(subset) == 0:
        raise ValueError(f"No records for skill_value={skill_value}")
    success_count = sum(1 for r in subset if int(r["success"]) == 1)
    return success_count / len(subset)


def laplace_smooth_prob(successes: int, trials: int) -> float:
    """
    Сглаживание Лапласа: P = (successes + 1) / (trials + 2)
    Чтобы избежать нулевых вероятностей
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("laplace_smooth_prob: invalid counts")
    return (successes + 1) / (trials + 2)


if __name__ == "__main__":
    # Данные: кандидаты с навыком Python и успешным наймом
    candidates = [
        {"python": 1, "success": 1},
        {"python": 1, "success": 0},
        {"python": 1, "success": 1},
        {"python": 1, "success": 1},
        {"python": 0, "success": 0},
        {"python": 0, "success": 1},
        {"python": 1, "success": 1},
        {"python": 0, "success": 1},
        {"python": 1, "success": 0},
        {"python": 1, "success": 1},
        {"python": 1, "success": 0},
        {"python": 1, "success": 1},
    ]

    print("=" * 60)
    print("📊 БАЙЕСОВСКИЙ СКОРИНГ КАНДИДАТОВ")
    print("=" * 60)

    counts = build_binary_counts(candidates, "success", "python")
    print(f"Частоты: {counts}")

    prior = prob_from_counts(counts["count_A"], counts["n"])
    likelihood = prob_conditional(counts["count_A_and_B"], counts["count_A"])
    evidence = prob_from_counts(counts["count_B"], counts["n"])

    posterior = bayes_posterior(prior, likelihood, evidence)

    print("-" * 40)
    print(f"P(успех) = {prior:.1%}")
    print(f"P(Python | успех) = {likelihood:.1%}")
    print(f"P(Python) = {evidence:.1%}")
    print("-" * 40)
    print(f"P(успех | Python) = {posterior:.1%}")
    print(f"📈 Знание Python повышает шансы в {posterior / prior:.1f} раз")

    p_success_no_skill = score_by_skill(candidates, 0)
    print(f"\nP(успех | нет Python) = {p_success_no_skill:.1%}")
    print("=" * 60)