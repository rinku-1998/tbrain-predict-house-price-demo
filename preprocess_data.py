import pandas as pd
import numpy as np
from scipy.special import boxcox1p
from util.geometry_util import twd97_to_lonlat
from util.json_util import load_json


def citydepart_to_label(city_depart, cd_to_label, cd_to_mean, city_mean):

    if city_depart in cd_to_label:
        return cd_to_label.get(city_depart)

    city = city_depart[:3]
    if city in cd_to_mean:
        return cd_to_mean.get(city)

    return city_mean


def preprocess(df_data: pd.DataFrame) -> pd.DataFrame:

    # 1. 移除不需要的欄位
    df_data.drop(labels=['使用分區', '備註'], axis=1, inplace=True)

    # 2. 轉換經緯度
    longtitudes = list()
    latitudes = list()
    for row, col in zip(df_data['橫坐標'], df_data['縱坐標']):

        lon, lat = twd97_to_lonlat(row, col)
        longtitudes.append(lon)
        latitudes.append(lat)

    df_data['經度'] = longtitudes
    df_data['緯度'] = latitudes

    # 3. 合併資料
    df_data['縣市鄉鎮市區'] = df_data['縣市'] + df_data['鄉鎮市區']
    df_data['總建物面積'] = df_data['主建物面積'] + df_data['附屬建物面積'] + df_data['陽台面積']
    df_data['總建物面積'].isnull().sum()
    df_data['has_parking'] = df_data['車位個數'].apply(lambda x: 1
                                                   if x >= 1 else 0)

    # 4. 資料轉換
    # 進行 Log Transform 之前先平移負數的維度
    offset_path = r'offset/col_to_offset.json'
    col_to_offset = load_json(offset_path)
    for col, offset in col_to_offset.items():
        df_data[col] = df_data[col] + abs(offset)

    # 根據訓練資料集套用偏移度>0.75資料 Box-Cox Transform
    convert_feats = [
        '附屬建物面積', '土地面積', '總建物面積', '建物面積', '主建物面積', '陽台面積', '車位面積', '車位個數',
        '經度', '橫坐標', '縱坐標', '緯度'
    ]
    lam = 0.15
    for feat in convert_feats:
        df_data[f'{feat}_transformed'] = boxcox1p(df_data[feat], lam)

    # 5. 數值資料分組
    # 移轉層次
    bins = [_ for _ in range(0, 60, 10)]
    labels = [_ for _ in range(5)]
    df_data['移轉層次_cat'] = pd.cut(df_data['移轉層次'],
                                 bins=bins,
                                 labels=labels,
                                 right=False)

    # 總樓層數
    bins = [_ for _ in range(0, 80, 10)]
    labels = [_ for _ in range(7)]
    df_data['總樓層數_cat'] = pd.cut(df_data['總樓層數'],
                                 bins=bins,
                                 labels=labels,
                                 right=False)

    # 分組屋齡
    bins = [_ for _ in range(0, 80, 10)]
    labels = [_ for _ in range(7)]
    df_data['屋齡_cat'] = pd.cut(df_data['屋齡'],
                               bins=bins,
                               labels=labels,
                               right=False)

    # 6. 資料編碼
    # 縣市
    city_path = r'encode/city.json'
    city_to_label = load_json(city_path)

    df_data['縣市_label'] = df_data['縣市'].map(city_to_label)
    df_data['縣市_label'].fillna(0.0, inplace=True)

    # 縣市鄉鎮市區
    cd_path = r'encode/city_depart.json'
    cd_to_label = load_json(cd_path)

    # 計算每個縣市下的鄉鎮市區編碼值
    df_cd = pd.DataFrame({
        'city_depart': [_ for _ in cd_to_label.keys()],
        'label': [_ for _ in cd_to_label.values()]
    })

    df_cd['city'] = df_cd['city_depart'].apply(lambda x: x[:3])
    df_cd['city_mean'] = df_cd.groupby('city')['label'].transform('mean')

    df_cd_mean = df_cd.loc[:, ['city', 'city_mean']]
    df_cd_mean.drop_duplicates(inplace=True)

    cd_to_mean = {row.city: row.city_mean for row in df_cd_mean.itertuples()}

    city_mean = df_cd['label'].mean()

    df_data['縣市鄉鎮市區_label'] = df_data['縣市鄉鎮市區'].apply(citydepart_to_label,
                                                      args=(cd_to_label,
                                                            cd_to_mean,
                                                            city_mean))
    df_data['縣市鄉鎮市區_label'].isnull().value_counts()

    df_data['縣市鄉鎮市區_label'].unique()

    # 主要用途
    mp_path = r'encode/main_purpose.json'
    mp_to_label = load_json(mp_path)

    df_data['主要用途_label'] = df_data['主要用途'].map(mp_to_label)
    df_data['主要用途_label'].fillna(mp_to_label.get('其他'), inplace=True)

    # 主要建材
    bm_path = r'encode/build_material.json'
    bm_to_label = load_json(bm_path)

    df_data['主要建材_label'] = df_data['主要建材'].map(bm_to_label)
    df_data['主要建材_label'].fillna(bm_to_label.get('其他'), inplace=True)

    # 建物型態
    bt_path = r'encode/build_type.json'
    bt_to_label = load_json(bt_path)

    df_data['建物型態_label'] = df_data['建物型態'].map(bt_to_label)

    bt_mean = np.mean([_ for _ in bt_to_label.values()])
    df_data['建物型態_label'].fillna(bt_mean, inplace=True)
    df_data['建物型態_label'].isnull().value_counts()

    return df_data
