from typing import List, Dict
import pandas as pd

from bitcoin.transformers.select_features import select_columns
from bitcoin.transformers.assign_features import assign_feature
from bitcoin.transformers.shift_features import shift_feature


if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform_df(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # select and assign
    df = select_columns(df)
    df = assign_feature(df)

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
    """
    Template code for testing the output of the block.
    """
    assert df is not None, "The output is undefined"
