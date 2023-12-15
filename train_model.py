import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer
from stacked_model import StackingAveragedModels
from typing import Tuple
from pathlib import Path
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')


def expm1_mape(y_true, y_pred):

    _y_true = np.expm1(y_true)
    _y_pred = np.expm1(y_pred)

    return mean_absolute_percentage_error(_y_true, _y_pred)


def prepare_data(df_data: pd.DataFrame) -> Tuple[pd.DataFrame]:

    # 1. 定義要使用的欄位
    used_cols = [
        '縣市_label', '縣市鄉鎮市區_label', '經度_transformed', '緯度_transformed',
        '土地面積_transformed', '總建物面積_transformed', '移轉層次_cat', '總樓層數_cat',
        '屋齡_cat', '主要用途_label', '主要建材_label', '建物型態_label', 'has_parking',
        '車位面積_transformed', 'cd_rprice', 'cdr_rprice',
        'train_station_num_transform', 'bus_stop_num_transform',
        'mrt_station_num_transform', 'bike_stop_num_transform',
        'elem_school_num_transform', 'jh_school_num_transform',
        'sh_school_num_transform', 'college_num_transform',
        'bank_num_transform', 'atm_num_transform', 'post_num_transform',
        'cvs_num_transform', 'wsclinic_num_transform',
        'chclinic_num_transform', 'dtclinic_num_transform',
        'healthctr_num_transform', 'hospital_num_transform'
    ]
    df_x = df_data.loc[:, used_cols]

    # 2. 目標值使用單價欄位
    df_y = df_data['單價']

    # 3. 取得數值(不要欄位名稱)
    # NOTE: 2023-10-23 先不分割，改用 K-Fold 來驗證
    train_x = df_x.values
    train_y = df_y.values

    return train_x, train_y


def train(df_data: pd.DataFrame):

    # 1. 準備訓練資料
    train_x, train_y = prepare_data(df_data)

    # 2. 測試模型效果
    # 找出每個模型的最佳參數(網格搜尋)
    # NOTE: 此範例就不再使用網格搜尋，直接用先前搜尋好的參數來建立模型
    print('Testing models')
    seed = 1
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    mape_scorer = make_scorer(expm1_mape, greater_is_better=False)

    # 建立 XGBoost 模型
    xgb_params = {
        'colsample_bytree': 0.5,
        'gamma': 0.0008,
        'learning_rate': 0.097,
        'max_depth': 7,
        'min_child_weight': 1,
        'n_estimators': 400,
        'reg_alpha': 0.00262,
        'reg_lambda': 0.012,
        'subsample': 0.7
    }
    xgb_model = XGBRegressor(objective='reg:squarederror',
                             **xgb_params,
                             random_state=0)
    xgb_scores = cross_val_score(xgb_model,
                                 X=train_x,
                                 y=train_y,
                                 cv=kf.split(train_x, train_y),
                                 n_jobs=4,
                                 scoring=mape_scorer,
                                 verbose=1)

    print('Model: XGBoost')
    print(f'Scores: {xgb_scores}')
    print(f'Mean: {xgb_scores.mean()}')
    print(f'Standard: {xgb_scores.std()}')

    # 建立堆疊模型
    # BaseModel
    # Lasso(L1)
    lasso_params = {'alpha': 1e-10, 'max_iter': 1000}
    lasso_model = make_pipeline(RobustScaler(),
                                Lasso(**lasso_params, random_state=0))

    # KernelRidge(L2)
    kridge_params = {'alpha': 1e-07, 'coef0': 6, 'degree': 2, 'gamma': 0.003}
    kridge_model = make_pipeline(
        RobustScaler(), KernelRidge(kernel='polynomial', **kridge_params))

    # ElasticNet
    enet_params = {'alpha': 1e-07, 'l1_ratio': 0.9}
    enet_model = make_pipeline(RobustScaler(),
                               ElasticNet(**enet_params, random_state=0))

    # Gradient Boost
    gbr_params = {
        'learning_rate': 0.15,
        'max_depth': 5,
        'min_samples_leaf': 0.0007,
        'min_samples_split': 0.002,
        'n_estimators': 800,
        'subsample': 0.7
    }
    gbr_model = GradientBoostingRegressor(loss='huber',
                                          **gbr_params,
                                          random_state=0)

    # 建立堆疊模型
    base_models = (enet_model, gbr_model, kridge_model)
    meta_model = lasso_model
    stacked_model = StackingAveragedModels(base_models=base_models,
                                           meta_model=meta_model)
    stacked_scores = cross_val_score(stacked_model,
                                     X=train_x,
                                     y=train_y,
                                     cv=kf.split(train_x, train_y),
                                     n_jobs=4,
                                     verbose=1,
                                     scoring=mape_scorer)

    print('Model: Stacked Model')
    print(f'Scores: {stacked_scores}')
    print(f'Mean: {stacked_scores.mean()}')
    print(f'Standard: {stacked_scores.std()}')

    # 3. 訓練正式模型
    print('Training final models')
    # 訓練堆疊模型
    stacked_model.fit(train_x, train_y)
    stacked_preds = stacked_model.predict(train_x)
    _stacked_scores = expm1_mape(train_y, stacked_preds)
    print('Model: Stacked model')
    print(f'MAPE(Before log-transform restoration): {_stacked_scores}')

    # 訓練 XGBoost
    xgb_model.fit(train_x, train_y)
    xgb_preds = xgb_model.predict(train_x)
    _xgb_scores = expm1_mape(train_y, xgb_preds)
    print('Model: XGBoost')
    print(f'MAPE(Before log-transform restoration): {_xgb_scores}')

    # 4. 計算還原 + 合併後的分數
    _stacked_preds = np.expm1(stacked_preds)
    _xgb_preds = np.expm1(xgb_preds)
    _train_y = np.expm1(train_y)

    # 合併
    _blended_preds = _stacked_preds * 0.5 + _xgb_preds * 0.5
    _blended_score = mean_absolute_percentage_error(_train_y, _blended_preds)
    print(
        f'MAPE(After log-transform restoration + merge with 50/50): {_blended_score}'
    )

    # 5. 保存模型
    Path('output/model').mkdir(parents=True, exist_ok=True)
    joblib.dump(stacked_model, 'output/model/stack_model.json')
    xgb_model.save_model('output/model/xgb_model.json')
