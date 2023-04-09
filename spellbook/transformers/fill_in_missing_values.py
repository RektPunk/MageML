import pandas as pd
import math

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


X_COLS = [
    "Age",
    "Fare",
    "Parch",
    "Pclass",
    "SibSp",
    "Survived",
]


def select_number_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[X_COLS]


def fill_missing_values_with_median(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        values = sorted(df[col].dropna().tolist())
        median_age = values[math.floor(len(values) / 2)]
        df = df.assign(**{col: df[[col]].fillna(median_age)})
    return df


@transformer
def transform_df(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """
    # Specify your transformation logic here

    return fill_missing_values_with_median(select_number_columns(df))


@test
def test_output(df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, "The output is undefined"
