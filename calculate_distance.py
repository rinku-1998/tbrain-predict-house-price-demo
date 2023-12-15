import pandas as pd
import numpy as np
import re
from pathlib import Path
from util.json_util import save_json
from typing import Tuple


def rad2deg(radians):
    return radians * 180 / np.pi


def deg2rad(degrees):
    return degrees * np.pi / 180


def get_distance(point1, point2, unit='kilometers'):

    # 1. 取得經緯度座標
    latitude1, longitude1 = point1
    latitude2, longitude2 = point2

    # 2. 計算角度
    theta = longitude1 - longitude2

    # 3. 計算距離
    distance = 60 * 1.1515 * rad2deg(
        np.arccos((np.sin(deg2rad(latitude1)) * np.sin(deg2rad(latitude2))) +
                  (np.cos(deg2rad(latitude1)) * np.cos(deg2rad(latitude2)) *
                   np.cos(deg2rad(theta)))))

    # 4. 計算指定單位的距離
    unit = 'kilometers' if unit not in ('kilometers', 'miles') else unit

    if unit == 'miles':
        return np.round(distance, 2)

    return np.round(distance * 1.609344, 2)


def calculate_college(df: pd.DataFrame, external_dir: str,
                      save_dir: str) -> None:

    # 1. 讀取資料
    college_path = str(Path(external_dir, '大學基本資料.csv'))
    df_college = pd.read_csv(college_path)

    # 2. 取出需要的欄位
    df_college = df_college.loc[:, ['縣市名稱', '學校代碼', 'lat', 'lng']]
    df_college.drop_duplicates(inplace=True)

    # 3. 移除縣市前的數字
    df_college['縣市名稱'] = df_college['縣市名稱'].apply(lambda x: x[3:])

    # 4. 統一文字，大台轉小台
    df_college['縣市名稱'] = df_college['縣市名稱'].apply(
        lambda x: x.replace('臺', '台'))

    # 5. 計算距離大學的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_college[df_college['縣市名稱'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 6. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'college.json')))


def calculate_bus_stop(df: pd.DataFrame, external_dir: str,
                       save_dir: str) -> None:

    # 1. 讀取資料
    bus_path = str(Path(external_dir, '公車站點資料.csv'))
    df_bus = pd.read_csv(bus_path)

    # 2. 取出需要的欄位
    df_bus = df_bus.loc[:, ['縣市', '站點UID', 'lat', 'lng']]
    df_bus.drop_duplicates(inplace=True)
    df_bus.head()

    # 3. 統一文字，大台轉小台
    df_bus['縣市'] = df_bus['縣市'].apply(lambda x: x.replace('臺', '台'))
    df_bus.head()

    # 4. 計算到公車站的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = [row.緯度, row.經度]

        distances = list()
        df_core = df_bus[df_bus['縣市'] == row.縣市]
        for i, idx in enumerate(df_core.index):
            point2 = [df_bus.iloc[idx, 2], df_bus.iloc[idx, 3]]
            dist = get_distance(point1, point2)
            distances.append(dist)
        distances_list.append(distances)

    # 5. 存檔公車站距離
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'bus_stop.json')))


def extract_city(text) -> str:

    # 1. 比對郵遞區號位址
    matches = [_ for _ in re.finditer(r'[\d]+', text)]

    # 2. 擷取縣市
    # 如果沒有開頭不是郵遞區號就直接擷取前三個字
    if not matches:
        return text[:3]

    if matches[0].start() != 0:
        return ''

    idx_start = matches[0].end()

    return text[idx_start:idx_start + 3]


def calculate_train_station(df: pd.DataFrame, external_dir: str,
                            save_dir: str) -> None:

    # 1. 讀取資料
    train_path = str(Path(external_dir, '火車站點資料.csv'))
    df_train = pd.read_csv(train_path)

    # 2. 取出需要的欄位
    df_train = df_train.loc[:, ['站點地址', '站點ID', 'lat', 'lng']]
    df_train.drop_duplicates(inplace=True)

    df_train['縣市'] = df_train['站點地址'].apply(extract_city)

    # 3. 統一文字，大台轉小台
    df_train['縣市'] = df_train['縣市'].apply(lambda x: x.replace('臺', '台'))
    df_train.head()

    # 4. 計算到火車站的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_train[df_train['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 5. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'train_station.json')))


def calculate_financial(df: pd.DataFrame, external_dir: str,
                        save_dir: str) -> None:

    # 1. 讀取資料
    bank_path = str(Path(external_dir, '金融機構基本資料.csv'))
    df_bank = pd.read_csv(bank_path)
    df_bank.head()

    # 2. 取出需要的欄位
    df_bank = df_bank.loc[:, ['地址', '銀行代號', '分支機構代號', 'lat', 'lng']]
    df_bank.drop_duplicates(inplace=True)
    df_bank.head()

    # 3. 擷取縣市
    df_bank['縣市'] = df_bank['地址'].apply(lambda x: x[:3])

    # 4. 計算到金融機構的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_bank[df_bank['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 5. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'bank.json')))


def calculate_cvs(df: pd.DataFrame, external_dir: str, save_dir: str) -> None:

    # 1. 讀取資料
    cvs_path = str(Path(external_dir, '便利商店.csv'))
    df_cvs = pd.read_csv(cvs_path)

    # 2. 取出需要的欄位
    df_cvs = df_cvs.loc[:, ['分公司地址', '公司名稱', '分公司名稱', 'lat', 'lng']]
    df_cvs.drop_duplicates(inplace=True)

    # 3. 取出縣市欄位
    df_cvs['縣市'] = df_cvs['分公司地址'].apply(lambda x: x[:3])

    # 統一文字，大台轉小台
    df_cvs['縣市'] = df_cvs['縣市'].apply(lambda x: x.replace('臺', '台'))

    # 4. 新舊縣市轉換
    city_map = {'台北縣': '新北市', '桃園縣': '桃園市', '高雄縣': '高雄市', '台南縣': '台南市'}

    df_cvs['縣市'] = df_cvs['縣市'].map(lambda x: city_map[x]
                                    if x in city_map else x)

    # 5. 計算到便利商店的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_cvs[df_cvs['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'cvs.json')))


def calculate_senior_high_school(df: pd.DataFrame, external_dir: str,
                                 save_dir: str) -> None:

    # 1. 讀取資料
    shschool_path = str(Path(external_dir, '高中基本資料.csv'))
    df_shschool = pd.read_csv(shschool_path)

    # 2. 取出需要的欄位
    df_shschool = df_shschool.loc[:, ['縣市名稱', '學校代碼', 'lat', 'lng']]
    df_shschool.drop_duplicates(inplace=True)

    # 3. 統一文字，大台轉小台
    df_shschool['縣市名稱'] = df_shschool['縣市名稱'].apply(
        lambda x: x.replace('臺', '台'))
    df_shschool['縣市名稱'].unique()

    # 4. 計算到高中的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_shschool[df_shschool['縣市名稱'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 5. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'senior_high_school.json')))


def calculate_elementary_school(df: pd.DataFrame, external_dir: str,
                                save_dir: str) -> None:

    # 1. 讀取資料
    elmschool_path = str(Path(external_dir, '國小基本資料.csv'))
    df_elmschool = pd.read_csv(elmschool_path)

    # 2. 取出需要的欄位
    df_elmschool = df_elmschool.loc[:, ['縣市名稱', '學校代碼', 'lat', 'lng']]
    df_elmschool.drop_duplicates(inplace=True)\

    # 3. 統一文字，大台轉小台
    df_elmschool['縣市名稱'] = df_elmschool['縣市名稱'].apply(
        lambda x: x.replace('臺', '台'))

    # 4. 計算到國小的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_elmschool[df_elmschool['縣市名稱'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 5. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'elementary_school.json')))


def calculate_junior_high_school(df: pd.DataFrame, external_dir: str,
                                 save_dir: str) -> None:

    # 1. 讀取資料
    jhschool_path = str(Path(external_dir, '國中基本資料.csv'))
    df_jhschool = pd.read_csv(jhschool_path)

    # 2. 取出需要的欄位
    df_jhschool = df_jhschool.loc[:, ['縣市名稱', '學校代碼', 'lat', 'lng']]
    df_jhschool.drop_duplicates(inplace=True)

    # 3. 統一文字，大台轉小台
    df_jhschool['縣市名稱'] = df_jhschool['縣市名稱'].apply(
        lambda x: x.replace('臺', '台'))
    df_jhschool['縣市名稱'].unique()

    # 4. 計算到國中的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_jhschool[df_jhschool['縣市名稱'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'junior_high_school.json')))


def calculate_mrt_station(df_data: pd.DataFrame, external_dir: str,
                          save_dir: str) -> None:

    # 1. 讀取資料
    mrt_path = str(Path(external_dir, '捷運站點資料.csv'))
    df_mrt = pd.read_csv(mrt_path)

    # 2. 取出需要的欄位
    df_mrt = df_mrt.loc[:, ['站點地址', '站點UID', 'lat', 'lng']]
    df_mrt.drop_duplicates(inplace=True)

    # 3. 計算到捷運站的距離
    distances_list = list()
    for row in df_data.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        for lat, lng in zip(df_mrt['lat'], df_mrt['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 4. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df_data.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'mrt_station.json')))


def calculate_post(df: pd.DataFrame, external_dir: str, save_dir: str) -> None:

    # 1. 讀取資料
    post_path = str(Path(external_dir, '郵局據點資料.csv'))
    df_post = pd.read_csv(post_path)

    # 2. 取出需要的欄位
    df_post = df_post.loc[:, ['局址', '電腦局號', 'lat', 'lng']]
    df_post.drop_duplicates(inplace=True)

    # 3. 取出縣市
    df_post['縣市'] = df_post['局址'].apply(lambda x: x[:3])

    # 4. 計算到郵局的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_post[df_post['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 5. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'distance/post.json')))


def calculate_bike(df: pd.DataFrame, external_dir: str, save_dir: str) -> None:

    # 1. 讀取資料
    bike_path = str(Path(external_dir, '腳踏車站點資料.csv'))
    df_bike = pd.read_csv(bike_path)

    # 取出需要的欄位
    df_bike = df_bike.loc[:, ['縣市', '站點UID', 'lat', 'lng']]
    df_bike.drop_duplicates(inplace=True)

    # 3. 計算到腳踏車站的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_bike[df_bike['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 4. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'bike.json')))


def categorize_medical(external_dir: str) -> Tuple[pd.DataFrame]:

    # 1. 讀取資料
    medical_path = str(Path(external_dir, '醫療機構基本資料.csv'))
    df_medical = pd.read_csv(medical_path)

    # 2. 取出需要的欄位
    df_medical = df_medical.loc[:, ['型態別', '縣市鄉鎮', '機構代碼', 'lat', 'lng']]
    df_medical.drop_duplicates(inplace=True)

    # 3. 擷取縣市欄位
    df_medical['縣市'] = df_medical['縣市鄉鎮'].apply(lambda x: x[:3])

    # 4. 統一文字，大台轉小台
    df_medical['縣市'] = df_medical['縣市'].apply(lambda x: x.replace('臺', '台'))

    # 5. 過濾不使用的醫療院所類別
    unused_medical = ('精神科醫院', '捐血站', '病理中心', '慢性醫院', '捐血中心', '中醫醫院', '專科醫院',
                      '其他醫療機構', '精神科教學醫院', '牙醫醫院')
    df_medical = df_medical[~df_medical['型態別'].isin(unused_medical)]

    # 6. 更新醫療院所類別名稱
    clinic_map = {
        '西醫診所': '西醫診所',
        '牙醫一般診所': '牙醫診所',
        '西醫專科診所': '西醫診所',
        '牙醫診所': '牙醫診所',
        '中醫診所': '中醫診所',
        '中醫一般診所': '中醫診所',
        '西醫醫務室': '西醫診所',
        '衛生所': '衛生所',
        '醫院': '醫院',
        '綜合醫院': '醫院',
        '中醫專科診所': '中醫診所',
        '牙醫專科診所': '牙醫診所'
    }

    df_medical['型態別'] = df_medical['型態別'].map(clinic_map)

    # 7. 分割醫療院所資料
    df_wsclinic = df_medical[df_medical['型態別'] == '西醫診所']
    df_chclinic = df_medical[df_medical['型態別'] == '中醫診所']
    df_dtclinic = df_medical[df_medical['型態別'] == '牙醫診所']
    df_healthctr = df_medical[df_medical['型態別'] == '衛生所']
    df_hospital = df_medical[df_medical['型態別'] == '醫院']

    return df_wsclinic, df_chclinic, df_dtclinic, df_healthctr, df_hospital


def calculate_western_clinic(df: pd.DataFrame, df_wsclinic: pd.DataFrame,
                             save_dir: str) -> None:

    # 1. 計算距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_wsclinic[df_wsclinic['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 2. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, '/western_clinic.json')))


def calculate_chinese_clinic(df: pd.DataFrame, df_chclinic: pd.DataFrame,
                             save_dir: str) -> None:

    # 1. 計算距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_chclinic[df_chclinic['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 2. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'chinese_clinic.json')))


def calculate_dental(df: pd.DataFrame, df_dtclinic: pd.DataFrame,
                     save_dir: str) -> None:

    # 1. 計算距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_dtclinic[df_dtclinic['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 2. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'dental_clinic.json')))


def calculate_health_center(df: pd.DataFrame, df_healthctr,
                            save_dir: str) -> None:

    # 1. 計算距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_healthctr[df_healthctr['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 2. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'health_center.json')))


def calculate_hospital(df: pd.DataFrame, df_hospital: pd.DataFrame,
                       save_dir: str) -> None:

    # 1. 計算距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_hospital[df_hospital['縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 2. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'hospital.json')))


def calculate_atm(df: pd.DataFrame, external_dir: str, save_dir: str):

    # 1. 讀取資料
    atm_path = str(Path(external_dir, 'ATM資料.csv'))
    df_atm = pd.read_csv(atm_path)

    # 2. 移除重複資料
    df_atm = df_atm.copy()
    df_atm.drop_duplicates(inplace=True)

    # 3. 計算到ATM的距離
    distances_list = list()
    for row in df.itertuples():
        point1 = (row.緯度, row.經度)

        distances = list()
        df_core = df_atm[df_atm['裝設縣市'] == row.縣市]
        for lat, lng in zip(df_core['lat'], df_core['lng']):
            point2 = (lat, lng)
            dist = get_distance(point1, point2)
            distances.append(dist)

        distances_list.append(distances)

    # 4. 存檔
    id_to_distances = dict()
    for _id, distances in zip(df.ID, distances_list):
        id_to_distances[_id] = distances

    save_json(id_to_distances, str(Path(save_dir, 'atm.json')))


def calculate(df_data: pd.DataFrame, data_type: str,
              external_dir: str) -> None:

    # 1. 設定存檔路徑
    save_dir = f'distance_{data_type}'

    # NOTE: 計算物件到周圍機能設施的距離
    # 2. 教育機構
    # 大學
    calculate_college(df_data, external_dir, save_dir)

    # 高中
    calculate_senior_high_school(df_data, external_dir, save_dir)

    # 國中
    calculate_junior_high_school(df_data, external_dir, save_dir)

    # 國小
    calculate_elementary_school(df_data, external_dir, save_dir)

    # 3. 交通
    # 公車站
    calculate_bus_stop(df_data, external_dir, save_dir)

    # 火車站
    calculate_train_station(df_data, external_dir, save_dir)

    # 捷運站
    calculate_mrt_station(df_data, external_dir, save_dir)

    # 腳踏車
    calculate_bike(df_data, external_dir, save_dir)

    # 4. 生活
    # 金融機構
    calculate_financial(df_data, external_dir, save_dir)

    # 便利商店
    calculate_cvs(df_data, external_dir, save_dir)

    # 郵局
    calculate_post(df_data, external_dir, save_dir)

    # ATM
    calculate_atm(df_data, external_dir, save_dir)

    # 5. 醫療機構
    # 分類不同的醫療機構
    df_wsclinic, df_chclinic, df_dtclinic, df_healthctr, df_hospital = categorize_medical(
        external_dir)

    # 西醫診所
    calculate_western_clinic(df_data, df_wsclinic, save_dir)

    # 中醫診所
    calculate_chinese_clinic(df_data, df_chclinic, save_dir)

    # 牙醫診所
    calculate_dental(df_data, df_dtclinic, save_dir)

    # 衛生所
    calculate_health_center(df_data, df_healthctr, save_dir)

    # 醫院
    calculate_hospital(df_data, df_hospital, save_dir)


if __name__ == '__main__':

    # 1. 讀取檔案
    data_path = r'dataset/train_data.csv'
    df_data = pd.read_csv(data_path)

    # 2. 計算物件與外部機能的距離
    external_dir = r'official_dataset/30_Training Dataset_V2/external_data'
    calculate(df_data, external_dir)
