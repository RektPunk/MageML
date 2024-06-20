from mage_ai.io.file import FileIO
import pandas as pd

if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_file(df: pd.DataFrame) -> None:
    filepath = "output.csv"
    FileIO().export(df, filepath)
