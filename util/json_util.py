import json
from pathlib import Path
from typing import Optional, Union


def load_json(json_path: str) -> Optional[Union[list, dict]]:
    """讀取 JSON 檔案

    Args:
        json_path (str): 檔案路徑

    Returns:
        Optional[Union[list, dict]]: 資料
    """

    # 1. 檢查檔案是否存在
    if not Path(json_path).exists():
        return None

    # 2. 讀取檔案
    data = None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_json(data: any, save_path: str) -> None:
    """儲存 JSON 檔案

    Args:
        data (any): 資料
        save_path (str): 儲存路徑ＦＦ
    """

    # 1. 檢查存擋路徑的上層路徑是否存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # 2. 寫入檔案
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
