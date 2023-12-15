import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor


def predict(df_test: str):

    # 1. 載入模型
    # 堆疊模型
    stacked_model = joblib.load(r'output/model/stack_model.json')

    # XGBoost
    xgb_model = XGBRegressor()
    xgb_model.load_model(r'output/model/xgb_model.json')


    # 2. 篩選欄位
    used_cols = [
        '縣市_label',
        '縣市鄉鎮市區_label',
        '經度_transformed',
        '緯度_transformed',  # 地理資訊(處理後)
        '土地面積_transformed',
        '總建物面積_transformed',  # 面積(處理後)
        '移轉層次_cat',
        '總樓層數_cat',
        '屋齡_cat',  # 物件狀況(處理後)
        '主要用途_label',
        '主要建材_label',
        '建物型態_label',  # 物件狀況(處理後)
        'has_parking',
        '車位面積_transformed',  # 車位
        'cd_rprice',
        'cdr_rprice',  # 實價登錄參考資料
        'train_station_num_transform',
        'bus_stop_num_transform',
        'mrt_station_num_transform',
        'bike_stop_num_transform',
        'elem_school_num_transform',
        'jh_school_num_transform',
        'sh_school_num_transform',
        'college_num_transform',
        'bank_num_transform',
        'atm_num_transform',
        'post_num_transform',
        'cvs_num_transform',
        'wsclinic_num_transform',
        'chclinic_num_transform',
        'dtclinic_num_transform',
        'healthctr_num_transform',
        'hospital_num_transform'
    ]
    df_x = df_test.loc[:, used_cols]

    # 3. 從 DataFrame 取出資料
    test_x = df_x.values

    # 4. 預測堆疊模型結果
    stacked_preds = stacked_model.predict(test_x)

    # 5. 預測 XGBoost 模型結果
    xgb_preds = xgb_model.predict(test_x)

    # 6. 還原對數變換結果
    _stacked_preds = np.expm1(stacked_preds)
    _xgb_preds = np.expm1(xgb_preds)

    # 7. 合併結果
    _preds = _stacked_preds * 0.5 + _xgb_preds * 0.5

    # 8. 存檔
    submission = {'ID': df_test['ID'].values, 'predicted_price': _preds}
    df_submission = pd.DataFrame(submission)
    df_submission.head()

    Path('output/submission').mkdir(exist_ok=True)
    df_submission.to_csv(r'output/submission/submission.csv', index=None)


if __name__ == '__main__':
    predict()
