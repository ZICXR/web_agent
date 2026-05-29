import math

def compute_entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs)


def calculate_entropy(response):
    results = []
    tokens = response.choices[0].logprobs.content
    if tokens is None:
        return {"content": "", "results": []}
    content = ""
    # token 是选择的最大的  top_logprobs 是全部的
    for token_info in tokens:  # 遍历所有token的可能取值

        token = token_info.token  # 当前生成 token

        probs = []
        top_k = []
        
        for t in token_info.top_logprobs:
            p = math.exp(t.logprob)
            dic = {"token": t.token, "p": p}
            top_k.append(dic)  # 保存各个
            probs.append(p)
        # 归一化
        Z = sum(probs)
        probs = [p / Z for p in probs]

        entropy = compute_entropy(probs)

        results.append({
            "token": token,
            "entropy": entropy,
            "top_k": top_k
        })
        content += token
    return {"content": content, "results": results}


def sliding_window_stats(results, window_size=10):
    stats = []

    entropies = [r["entropy"] for r in results]

    for i in range(len(results)):
        start = max(0, i - window_size + 1)
        window = entropies[start:i+1]

        mu = sum(window) / len(window)
        sigma = (sum((x - mu) ** 2 for x in window) / len(window)) ** 0.5

        stats.append({
            "step": i,
            "token": results[i]["token"],
            "entropy": results[i]["entropy"],
            "mu": mu,
            "sigma": sigma
        })

    return stats


