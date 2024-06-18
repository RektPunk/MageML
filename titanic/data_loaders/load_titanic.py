import io
import pandas as pd
import requests
from pandas import DataFrame
from titanic.utils.variables import TITANIC_DATA_URL


if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api() -> DataFrame:
    """
    Load titanic data
    """
    _response = requests.get(TITANIC_DATA_URL)
    return pd.read_csv(io.StringIO(_response.text), sep=",")


@test
def test_output(df: pd.DataFrame) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, "The output is undefined"