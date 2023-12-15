import pandas as pd
from pathlib import Path


def save_csv(df: pd.DataFrame, save_path: str) -> None:

    # 1. 檢查存擋路徑的資料夾是否存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # 2. 存擋
    df.to_csv(save_path, index=None)
