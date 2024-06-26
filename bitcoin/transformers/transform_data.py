from typing import List, Dict
import numpy as np
import pandas as pd

from utils.shift_feature import shift_feature


if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델에 사용할 컬럼, 컬럼의 rename rule을 미리 할당함
    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
    cols_dict: Dict[str, str] = {
        "ID": "ID",
        "target": "target",
        "_type": "_type",
        "hourly_market-data_coinbase-premium-index_coinbase_premium_gap": "coinbase_premium_gap",
        "hourly_market-data_coinbase-premium-index_coinbase_premium_index": "coinbase_premium_index",
        "hourly_market-data_funding-rates_all_exchange_funding_rates": "funding_rates",
        "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations": "long_liquidations",
        "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd": "long_liquidations_usd",
        "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations": "short_liquidations",
        "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations_usd": "short_liquidations_usd",
        "hourly_market-data_open-interest_all_exchange_all_symbol_open_interest": "open_interest",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio": "buy_sell_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "buy_volume",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "sell_volume",
        "hourly_network-data_addresses-count_addresses_count_active": "active_count",
        "hourly_network-data_addresses-count_addresses_count_receiver": "receiver_count",
        "hourly_network-data_addresses-count_addresses_count_sender": "sender_count",
    }
    df = df[cols_dict.keys()].rename(cols_dict, axis=1)
    return df


def _assign_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    변수간 차이와 차이의 음수, 양수 여부를 새로운 피쳐로 생성
    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
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


@transformer
def transform_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # select and assign
    df = _select_columns(df)
    df = _assign_feature(df)

    # shift
    conti_cols: List[str] = [
        _ for _ in df.columns if _ not in ["ID", "target", "_type"]
    ] + [
        "buy_sell_volume_ratio",
        "liquidation_diff",
        "liquidation_usd_diff",
        "volume_diff",
    ]
    shift_features = shift_feature(
        df=df, conti_cols=conti_cols, intervals=[_ for _ in range(1, 24)]
    )
    df = pd.concat([df, shift_features], axis=1)

    # imputation
    _target = df["target"]
    df = df.ffill().fillna(-999).assign(target=_target)

    # split for train
    train_df = df.loc[df["_type"] == "train"].drop(columns=["_type"])
    test_df = df.loc[df["_type"] == "test"].drop(columns=["_type"])

    return {
        "train_df": train_df,
        "test_df": test_df,
    }


@test
def test_output(df) -> None:
    assert df is not None, "The output is undefined"
