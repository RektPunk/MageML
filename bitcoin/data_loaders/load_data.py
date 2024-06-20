import os
import io
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb


if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(**kwargs) -> pd.DataFrame:
    """
    Template for loading data from API
    """
    # 파일 호출
    data_path: str = os.path.join("bitcoin", "data")
    train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv")).assign(
        _type="train"
    )  # train 에는 _type = train
    test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")).assign(
        _type="test"
    )  # test 에는 _type = test
    submission_df: pd.DataFrame = pd.read_csv(
        os.path.join(data_path, "test.csv")
    )  # ID, target 열만 가진 데이터 미리 호출
    df: pd.DataFrame = pd.concat([train_df, test_df], axis=0)

    # HOURLY_ 로 시작하는 .csv 파일 이름을 file_names 에 할딩
    file_names: List[str] = [
        f
        for f in os.listdir(data_path)
        if f.startswith("HOURLY_") and f.endswith(".csv")
    ]

    # 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
    file_dict: Dict[str, pd.DataFrame] = {
        f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f))
        for f in file_names
    }

    for _file_name, _df in tqdm(file_dict.items()):
        # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
        _rename_rule = {
            col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
            for col in _df.columns
        }
        _df = _df.rename(_rename_rule, axis=1)
        df = df.merge(_df, on="ID", how="left")

    return df


@test
def test_output(*df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, "The output is undefined"
