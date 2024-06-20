import os
import pandas as pd
from bitcoin.utils.variable import data_path

if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_sample_submission(**kwargs) -> pd.DataFrame:
    # 파일 호출

    submission_df: pd.DataFrame = pd.read_csv(
        os.path.join(data_path, "sample_submission.csv")
    )  # ID, target 열만 가진 데이터 미리 호출
    return submission_df


@test
def test_output(*df) -> None:
    assert df is not None, "The output is undefined"
