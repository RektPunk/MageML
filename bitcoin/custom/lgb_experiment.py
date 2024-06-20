import logging
from typing import Union, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb


if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def lgb_verify_score(*args, **kwargs):
    # train_test_split 으로 valid set, train set 분리
    train_df = args[0].get("train_df")
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_df.drop(["target", "ID"], axis=1),
        train_df["target"].astype(int),
        test_size=0.2,
        random_state=42,
    )

    # lgb dataset
    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)

    # lgb train
    params: Dict[str, Union[str, int, float]] = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": 4,
        "num_leaves": 50,
        "learning_rate": 0.05,
        "n_estimators": 30,
        "random_state": 42,
        "verbose": 0,
    }  # TODO: hyperparameter tuning, logging logic
    lgb_model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=valid_data,
    )

    # lgb predict
    y_valid_pred = lgb_model.predict(x_valid)
    y_valid_pred_class = np.argmax(y_valid_pred, axis=1)

    # score check
    accuracy = accuracy_score(y_valid, y_valid_pred_class)
    auroc = roc_auc_score(y_valid, y_valid_pred, multi_class="ovr")

    logging.info(f"acc: {accuracy}, auroc: {auroc}")
    return {"params": params}
