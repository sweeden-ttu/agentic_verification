import math


def entropy(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def kl(p, q):
    return sum(a * math.log2(a / b) for a, b in zip(p, q) if a > 0 and b > 0)


def posterior_mean(counts, alpha):
    n = sum(counts)
    k = len(counts)
    return [(c + alpha) / (n + alpha * k) for c in counts]


def log_marginal(counts, alpha):
    k = len(counts)
    n = sum(counts)
    return (math.lgamma(alpha * k) - math.lgamma(alpha * k + n)
            + sum(math.lgamma(alpha + c) - math.lgamma(alpha) for c in counts))


def decompose(count_map, alpha=0.5):
    models = list(count_map)
    k = len(next(iter(count_map.values())))
    post = {m: posterior_mean(count_map[m], alpha) for m in models}
    pbar = [sum(post[m][i] for m in models) / len(models) for i in range(k)]
    h_total = entropy(pbar)
    h_within = sum(entropy(post[m]) for m in models) / len(models)
    mi = max(0.0, h_total - h_within)
    pooled = [sum(count_map[m][i] for m in models) for i in range(k)]
    log_bf = sum(log_marginal(count_map[m], alpha) for m in models) - log_marginal(pooled, alpha)
    surprise = {}
    for m in models:
        others = [o for o in models if o != m]
        if not others:
            surprise[m] = 0.0
            continue
        q = [sum(post[o][i] for o in others) / len(others) for i in range(k)]
        surprise[m] = kl(post[m], q)
    return {
        "post": post,
        "pbar": pbar,
        "h_total": h_total,
        "h_within": h_within,
        "mi": mi,
        "meaningful_share": mi / h_total if h_total > 0 else 0.0,
        "log_bf": log_bf,
        "surprise": surprise,
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
