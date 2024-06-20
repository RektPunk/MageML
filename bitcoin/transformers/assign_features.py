import numpy as np
import pandas as pd


def assign_feature(df: pd.DataFrame) -> pd.DataFrame:
    # eda 에서 파악한 차이와 차이의 음수, 양수 여부를 새로운 피쳐로 생성
    df = df.assign(
        liquidation_diff=df["long_liquidations"] - df["short_liquidations"],
        liquidation_usd_diff=df["long_liquidations_usd"] - df["short_liquidations_usd"],
        volume_diff=df["buy_volume"] - df["sell_volume"],
        liquidation_diffg=np.sign(df["long_liquidations"] - df["short_liquidations"]),
        liquidation_usd_diffg=np.sign(
            df["long_liquidations_usd"] - df["short_liquidations_usd"]
        ),
        volume_diffg=np.sign(df["buy_volume"] - df["sell_volume"]),
        buy_sell_volume_ratio=df["buy_volume"] / (df["sell_volume"] + 1),
    )
    return df
