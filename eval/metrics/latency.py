def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * p)
    return values[idx]
