import numpy as np
import pandas as pd
import warnings
from scipy.stats import skew
from scipy.special import boxcox1p
from typing import Dict, List
from util.csv_util import save_csv
from util.geometry_util import twd97_to_lonlat
from util.json_util import save_json, load_json

warnings.filterwarnings('ignore')


def aggregate_feat(df: pd.DataFrame) -> pd.DataFrame:
    """合併特徵

    Args:
        df (pd.DataFrame): 房價資料集 DataFrame

    Returns:
        pd.DataFrame: 合併後的房價資料集 DataFrame
    """

    # 1. 合併縣市鄉鎮市區與地址
    df['縣市鄉鎮市區'] = df['縣市'] + df['鄉鎮市區']
    df['地址'] = df['縣市'] + df['鄉鎮市區'] + df['路名']

    # 2. 合併建物總面積
    df['總建物面積'] = df['主建物面積'] + df['附屬建物面積'] + df['陽台面積']

    # 3. 合併車位資料
    df['has_parking'] = df['車位個數'].apply(lambda x: 1 if x >= 1 else 0)

    return df


def transform_value(df: pd.DataFrame) -> pd.DataFrame:
    """轉換數值

    Args:
        df (pd.DataFrame): 房價資料集 DataFrame

    Returns:
        pd.DataFrame: 轉換後的房價資料集 DataFrame
    """

    # 1. 轉換單價(目標值)，使用 Log Transform
    df['單價'] = np.log1p(df['單價'])

    # 2. 轉換其他數值欄位
    # 計算數值型特徵的偏移情況
    numerical_cols = df.dtypes[df.dtypes != 'object'].index

    skew_feats = df[numerical_cols].apply(
        lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skew_feats})

    convert_cols: List[str] = list()
    for skew_col, skew_value in zip(skewness.index, skewness.values):
        if abs(skew_value) <= 0.75:
            continue

        if skew_col in ('總樓層數', '屋齡', '移轉層次'):
            continue

        convert_cols.append(skew_col)

    # 平移數值
    # NOTE: 做 Log 或 Box-Cox Transform 之前要先確保沒有負數，如果有負數要先平移
    col_to_offset: Dict[str, float] = dict()
    for col in convert_cols:

        if col in ('橫坐標', ):
            continue

        min_value = min(df[col])
        if min_value < 0:

            offset = abs(min_value)
            df[col] = df[col] + offset
            col_to_offset[col] = min_value

    # 儲存平移數值
    save_json(col_to_offset, 'offset/col_to_offset.json')

    # 數值型資料套用 Box-Cox Transform
    # NOTE: 偏移度 >0.75 的資料才需要作轉換，Lambda 使用 0.15
    lam = 0.15
    for col in convert_cols:
        df[f'{col}_transformed'] = boxcox1p(df[col], lam)

    return df


def group_by_bin(df: pd.DataFrame) -> pd.DataFrame:

    # 1. 分組移轉層次
    bins = [_ for _ in range(0, 60, 10)]
    labels = [_ for _ in range(5)]
    df['移轉層次_cat'] = pd.cut(df['移轉層次'], bins=bins, labels=labels, right=False)

    # 2. 分組總樓層數
    bins = [_ for _ in range(0, 80, 10)]
    labels = [_ for _ in range(7)]
    df['總樓層數_cat'] = pd.cut(df['總樓層數'], bins=bins, labels=labels, right=False)

    # 3. 分組屋齡
    bins = [_ for _ in range(0, 80, 10)]
    labels = [_ for _ in range(7)]
    df['屋齡_cat'] = pd.cut(df['屋齡'], bins=bins, labels=labels, right=False)

    return df


def citydepart_to_label(city_depart, cd_to_label, cd_to_mean, city_mean):

    if city_depart in cd_to_label:
        return cd_to_label.get(city_depart)

    city = city_depart[:3]
    if city in cd_to_mean:
        return cd_to_mean.get(city)

    return city_mean


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:

    # 1. 編碼縣市
    labels = df.groupby('縣市')['單價'].transform('mean')
    df['縣市_label'] = labels

    # 存擋縣市編碼值
    df_city_label = pd.DataFrame({'city': df['縣市'].values, 'label': labels})
    df_city_label.drop_duplicates(inplace=True)
    df_city_label.sort_values(by=['label'], ascending=False, inplace=True)

    city_to_label = {row.city: row.label for row in df_city_label.itertuples()}
    save_json(city_to_label, 'encode/city.json')

    # 2. 編碼縣市鄉鎮市區
    labels = df.groupby('縣市鄉鎮市區')['單價'].transform('mean')
    df['縣市鄉鎮市區_label'] = labels

    # 存擋縣市鄉鎮市區編碼值
    df_citydepart_label = pd.DataFrame({
        'citydepart': df['縣市鄉鎮市區'].values,
        'label': labels
    })
    df_citydepart_label.drop_duplicates(inplace=True)
    df_citydepart_label.sort_values(by=['label'],
                                    ascending=False,
                                    inplace=True)

    citydepart_to_label = {
        row.citydepart: row.label
        for row in df_citydepart_label.itertuples()
    }
    save_json(citydepart_to_label, 'encode/city_depart.json')

    # 3. 編碼主要用途
    labels = df.groupby('主要用途')['單價'].transform('mean')
    df['主要用途_label'] = labels

    # 存擋主要用途編碼值
    df_mp_label = pd.DataFrame({
        'main_purpose': df['主要用途'].values,
        'label': labels
    })
    df_mp_label.drop_duplicates(inplace=True)
    df_mp_label.sort_values(by=['label'], ascending=False, inplace=True)

    mp_to_label = {
        row.main_purpose: row.label
        for row in df_mp_label.itertuples()
    }
    save_json(mp_to_label, 'encode/main_purpose.json')

    # 4. 編碼主要建材
    labels = df.groupby('主要建材')['單價'].transform('mean')
    df['主要建材_label'] = labels

    # 存擋主要建材編碼值
    df_bm_label = pd.DataFrame({
        'build_material': df['主要建材'].values,
        'label': labels
    })
    df_bm_label.drop_duplicates(inplace=True)
    df_bm_label.sort_values(by=['label'], ascending=False, inplace=True)

    bm_to_label = {
        row.build_material: row.label
        for row in df_bm_label.itertuples()
    }
    save_json(bm_to_label, 'encode/build_material.json')

    # 5. 編碼建物型態
    labels = df.groupby('建物型態')['單價'].transform('mean')
    df['建物型態_label'] = labels

    # 存擋建物型態編碼值
    df_bt_label = pd.DataFrame({
        'build_type': df['建物型態'].values,
        'label': labels
    })
    df_bt_label.drop_duplicates(inplace=True)
    df_bt_label.sort_values(by=['label'], ascending=False, inplace=True)
    df_bt_label.head(10)

    bt_to_labels = {
        row.build_type: row.label
        for row in df_bt_label.itertuples()
    }
    save_json(bt_to_labels, 'encode/build_type.json')

    return df


def clean(df_price: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:

    # 1. 移除欄位
    # NOTE: 移除資料少很多的欄位
    df_price.drop(labels=['使用分區', '備註'], axis=1, inplace=True)

    # 2. 轉換經緯度
    longtitudes: List[float] = list()
    latitudes: List[float] = list()
    for row, col in zip(df_price['橫坐標'], df_price['縱坐標']):

        lon, lat = twd97_to_lonlat(row, col)
        longtitudes.append(lon)
        latitudes.append(lat)

    df_price['經度'] = longtitudes
    df_price['緯度'] = latitudes

    # 3. 移除離群值
    # NOTE: 台南市復國一路單價異常的物件
    df_price = df_price.drop(df_price[(df_price['縣市'] == '台南市')
                                      & (df_price['單價'] > 12)].index)

    # 4. 合併資料
    df_price = aggregate_feat(df_price)

    # 5. 資料轉換(Transform)
    df_price = transform_value(df_price)

    # 6. 數值資料分組並編碼
    df_price = group_by_bin(df_price)

    # 7. 資料編碼
    df_price = encode_categorical(df_price)

    # 8. 保留需要的欄位
    # used_cols = [
    #     'ID', '縣市', '鄉鎮市區', '路名', '經度', '緯度', '縣市_label', '縣市鄉鎮市區_label',
    #     '經度_transformed', '緯度_transformed', '土地面積_transformed',
    #     '總建物面積_transformed', '移轉層次_cat', '總樓層數_cat', '屋齡_cat', '主要用途_label',
    #     '主要建材_label', '建物型態_label', 'has_parking', '車位面積_transformed'
    # ]
    # if is_train:
    #     used_cols.append('單價')

    # df_eda = df_price.loc[:, used_cols]

    return df_price


if __name__ == '__main__':

    # 1. 讀取資料集
    df_price = pd.read_csv(
        'official_dataset/30_Training Dataset_V2/training_data.csv')

    # 2. EDA
    df_eda = clean(df_price, is_train=True)

    # 3. 存擋
    save_csv(df_eda, 'dataset/train_data.csv')
