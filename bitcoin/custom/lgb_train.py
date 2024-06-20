import numpy as np
import pandas as pd
import lightgbm as lgb


if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def lgb_train(submission_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    lgb train whole train dataset
    Args:
        submission_df (pd.DataFrame)
        *args: {train_df, test_df}, {params}
    Returns:
        pd.DataFrame
    """
    train_df = args[0].get("train_df")
    test_df = args[0].get("test_df")
    params = args[1].get("params")
    assert train_df is not None, "args does not have key named 'train_df'"
    assert test_df is not None, "args does not have key named 'test_df'"
    assert params is not None, "args does not have key named 'params'"

    x_train = train_df.drop(["target", "ID"], axis=1)
    y_train = train_df["target"].astype(int)
    train_data = lgb.Dataset(x_train, label=y_train)
    lgb_model = lgb.train(
        params=params,
        train_set=train_data,
    )
    # TODO: model 저장하는 코드

    y_test_pred = lgb_model.predict(test_df.drop(["target", "ID"], axis=1))
    y_test_pred_class = np.argmax(y_test_pred, axis=1)
    submission_df = submission_df.assign(target=y_test_pred_class)
    return submission_df
