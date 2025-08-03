from mpmath import mp

def mp_logsumexp(values):
    if not values:
        return mp.ninf
    max_val = max(values)
    sum_exp = mp.mpf(0)
    for v in values:
        sum_exp += mp.exp(v - max_val)
    return max_val + mp.log(sum_exp)