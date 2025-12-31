# feature_engineering.py
import pandas as pd

from utils import EPS


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered interaction features:
      - power_to_weight
      - torque_to_weight
      - car_age_ratio
      - engine_efficiency
    """
    # power/torque vs weight
    if {"max_power_value", "gross_weight"}.issubset(df.columns):
        df["power_to_weight"] = df["max_power_value"] / (df["gross_weight"] + EPS)

    if {"max_torque_value", "gross_weight"}.issubset(df.columns):
        df["torque_to_weight"] = df["max_torque_value"] / (df["gross_weight"] + EPS)

    # car age vs policy tenure
    if {"age_of_car", "policy_tenure"}.issubset(df.columns):
        df["car_age_ratio"] = df["age_of_car"] / (df["policy_tenure"] + 1.0)

    # engine "efficiency"
    if {"displacement", "max_power_value"}.issubset(df.columns):
        df["engine_efficiency"] = df["displacement"] / (df["max_power_value"] + EPS)

    return df
