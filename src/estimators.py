def compute_ips(df):
    assert {"model_prob", "ad_prob", "ad_revenue"}.issubset(df.columns)
    return (df["model_prob"] / df["ad_prob"] * df["ad_revenue"]).mean()


def compute_ncis(logs, cap=1000):
    assert {"model_prob", "ad_prob", "ad_revenue"}.issubset(logs.columns)
    df = logs.copy()

    df["cap"] = cap
    df["probs"] = df["model_prob"] / df["ad_prob"]

    df["min_probs_or_cap"] = df[["cap", "probs"]].min(axis=1)

    ncis_num = (df["ad_revenue"] * df["min_probs_or_cap"]).mean()
    ncis_denom = df["min_probs_or_cap"].mean()

    return ncis_num / ncis_denom
