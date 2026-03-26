import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["bytes_total"] = df["sbytes"] + df["dbytes"]
    df["pkts_total"] = df["spkts"] + df["dpkts"]
    df["byte_ratio"] = df["sbytes"] / (df["dbytes"] + 1.0)
    df["pkt_ratio"] = df["spkts"] / (df["dpkts"] + 1.0)
    df["load_ratio"] = df["sload"] / (df["dload"] + 1.0)
    df["ttl_diff"] = (df["sttl"] - df["dttl"]).abs()
    df["jit_total"] = df["sjit"] + df["djit"]
    df["mean_size_total"] = df["smeansz"] + df["dmeansz"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df