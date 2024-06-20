import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from titanic.utils.variables import X_COLS, Y_COLS


if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100


class NeuralNetwork(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid()
        )

    def forward(self, x):
        probs = self.linear_relu_stack(x)
        return probs


def _model_save():
    """
    save model some place
    """
    pass


@custom
def torch_train(df: pd.DataFrame, *args, **kwargs):
    """
    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # some model code

    x_train = df[X_COLS]
    y_train = df[Y_COLS]
    x_train_tensor = torch.Tensor(x_train.to_numpy())
    y_train_tensor = torch.Tensor(y_train.to_numpy())

    x_train_dataloader = DataLoader(x_train_tensor, batch_size=BATCH_SIZE)
    y_train_dataloader = DataLoader(y_train_tensor, batch_size=BATCH_SIZE)

    some_nn = NeuralNetwork(x_train.shape[1])
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(some_nn.parameters(), lr=LEARNING_RATE)

    for _ in range(EPOCHS):
        for (_, _x), (_, _y) in zip(
            enumerate(x_train_dataloader), enumerate(y_train_dataloader)
        ):
            pred = some_nn(_x)
            loss = loss_fn(pred, _y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if _ % 10 == 0:
            #     _loss = loss.item()
            #     print(f"loss: {_loss:>7f}")

    _model_save()
    _pred = some_nn(x_train_tensor).round().detach().numpy()
    df = df.assign(Survived_predict=_pred.astype(int))
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
