import pandas as pd
import numpy as np
from typing import List


def merge_rp(df: pd.DataFrame) -> pd.DataFrame:

    # 1. 讀取資料
    cdrmean_path = r'real_price/cdrmean.csv'
    df_cdrmean = pd.read_csv(cdrmean_path)
    df_cdrmean.head()

    df_cdrmean['road'].fillna('', inplace=True)

    # 2. 從實價登錄資料裡面找接近的物件做為參考值
    cdr_rprices: List[float] = list()
    cd_rprices: List[float] = list()
    for row in df.itertuples():

        # 1. 從所有欄位依序尋找，挑最符合者
        # 縣市 (city)
        df_cmatch = df_cdrmean[df_cdrmean['city'] == row.縣市]

        # 鄉鎮市區 (city, deaprt)
        df_cdmatch = df_cmatch[df_cmatch['depart'] == row.鄉鎮市區]

        # 路名 (city, deaprt, road)
        df_cdrmatch = df_cdmatch[df_cdmatch['road'].str.startswith(row.路名)]

        # 移轉層次 (city, depart, road, level)
        level_dv10 = row.移轉層次 // 10
        df_cdrlmatch = df_cdrmatch[df_cdrmatch['level'] // 10 == level_dv10]

        # 屋齡 (city, depart, road, age)
        age_dv10 = row.屋齡 // 10
        df_cdramatch = df_cdrmatch[df_cdrmatch['age'] // 10 == age_dv10]

        # 移轉層次+屋齡 (city, depart, road, level, age)
        df_cdrlamatch = df_cdrmatch[(df_cdrmatch['level'] == row.移轉層次)
                                    & (df_cdrmatch['age'] == row.屋齡)]

        # 2. 從最多符合的比對回去
        cdr_rprice = None

        # 移轉層次+屋齡 (city, depart, road, level, age)
        if not df_cdrlamatch.empty:
            cdr_rprice = df_cdrlamatch['mean'].mean()

        # 屋齡 (city, depart, road, age)
        if (cdr_rprice is None) and (not df_cdramatch.empty):
            cdr_rprice = df_cdramatch['mean'].mean()

        # 移轉層次 (city, depart, road, level)
        if (cdr_rprice is None) and (not df_cdrlmatch.empty):
            cdr_rprice = df_cdrlmatch['mean'].mean()

        # 路名 (city, depart, road)
        if (cdr_rprice is None) and (not df_cdrlmatch.empty):
            cdr_rprice = df_cdrlmatch['mean'].mean()

        # 鄉鎮市區 (city, deaprt)
        if (cdr_rprice is None) and (not df_cdmatch.empty):
            cdr_rprice = df_cdmatch['mean'].mean()

        cdr_rprices.append(cdr_rprice)
        cd_rprice = df_cdmatch['mean'].mean()
        cd_rprices.append(cd_rprice)

    df['cdr_rprice'] = cdr_rprices
    df['cd_rprice'] = cd_rprices

    # 3. 除以縮放係數 + 對數變換
    factor = 204083.8070980369
    df['cdr_rprice'] = np.log1p(df['cdr_rprice'] / factor)
    df['cd_rprice'] = np.log1p(df['cd_rprice'] / factor)

    return df

