"""Echo-score fixed text. Never execute the candidate action.

This adapter provides a reduced logprob profile, not full-vocabulary PBGP.
Only tokens wholly inside the continuation are included. Reject boundaries
that split tokens so context edits cannot silently change the scored action.
"""
import json
import math
import statistics
import urllib.error
import urllib.request

BASE_URL = "https://hermes.ai.unturf.com/v1"


def api_request(path, body=None, base_url=BASE_URL):
    req = urllib.request.Request(
        base_url + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer pbgp-public-capability-probe",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            return {"status": response.status, "response": json.loads(response.read())}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "error": exc.read().decode(errors="replace")[:3000]}
    except Exception as exc:
        return {"error": str(exc)}


def extract_continuation(response, prefix, continuation):
    choice = response["choices"][0]
    text = prefix + continuation
    if choice["text"] != text:
        raise ValueError("Echo does not equal the requested text")
    lp = choice["logprobs"]
    tokens, values, offsets = lp["tokens"], lp["token_logprobs"], lp["text_offset"]
    if not (len(tokens) == len(values) == len(offsets)):
        raise ValueError("Token arrays are not aligned")
    if not offsets or offsets[0] != 0 or any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Invalid text offsets")
    if "".join(tokens) != text:
        raise ValueError("Token text does not reconstruct the ASCII probe")
    boundary = len(prefix)
    if boundary not in offsets:
        raise ValueError("Context/continuation boundary crosses a token")
    start = offsets.index(boundary)
    kept = values[start:]
    if not kept or any(v is None or not math.isfinite(v) or v > 0 for v in kept):
        raise ValueError("Missing or invalid continuation logprobs")
    if "".join(tokens[start:]) != continuation:
        raise ValueError("Scored tokens differ from the fixed continuation")
    return {"tokens": tokens[start:], "logprobs": kept,
            "offsets": [o - boundary for o in offsets[start:]],
            "num_tokens": len(kept), "mean_nll": -statistics.mean(kept),
            "logprob_std": statistics.pstdev(kept),
            "mean_abs_step": statistics.mean(abs(b-a) for a, b in zip(kept, kept[1:]))
            if len(kept) > 1 else 0.0}


def compare_profiles(a, b):
    if a["tokens"] != b["tokens"]:
        raise ValueError("Fixed continuation tokenization changed across contexts")
    x, y = a["logprobs"], b["logprobs"]
    # Equal-size empirical 1D Wasserstein distance; no sampling or fitted bins.
    wasserstein = statistics.mean(abs(u-v) for u, v in zip(sorted(x), sorted(y)))
    # Fixed histogram edges, with all logprobs below -32 in the tail bin.
    def hist(values):
        counts = [0] * 17
        for v in values:
            counts[0 if v < -32 else min(16, int((v + 32) / 2) + 1)] += 1
        return [c / len(values) for c in counts]
    p, q = hist(x), hist(y)
    m = [(u+v)/2 for u, v in zip(p, q)]
    def kl(first, second):
        return sum(u * math.log2(u/v) for u, v in zip(first, second) if u)
    return {"wasserstein_logprobs": wasserstein,
            "js_distance_bits": math.sqrt((kl(p, m) + kl(q, m))/2),
            "mean_abs_token_delta": statistics.mean(abs(u-v) for u, v in zip(x, y)),
            "mean_nll_delta": b["mean_nll"] - a["mean_nll"]}


def auc(labels, scores):
    positives = [s for label, s in zip(labels, scores) if label == 1]
    negatives = [s for label, s in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    return sum((p > n) + .5 * (p == n) for p in positives for n in negatives) / (len(positives)*len(negatives))
