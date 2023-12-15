import pandas as pd
from util.csv_util import save_csv
from util.json_util import load_json
from typing import Dict, List
from pathlib import Path
from scipy.stats import skew
from scipy.special import boxcox1p


def count_num(ids: List[str],
              id_to_distances: Dict[str, List[float]],
              threshold: int = 1) -> List[int]:

    # 1. 計算數量
    nums = list()
    for _id in ids:
        distances = id_to_distances[_id]
        num = sum(_ <= threshold for _ in distances)
        nums.append(num)

    return nums


def num_to_has(nums: List[int]) -> List[int]:
    return [int(bool(_)) for _ in nums]


def merge_ext(df_data: pd.DataFrame, distance_dir: str) -> pd.DataFrame:

    # 1. 取得資料集的 ID
    ids = df_data['ID'].values

    # 2. 計算交通外部資料距離
    # 火車站(數量+有無)
    train_path = str(Path(distance_dir, 'train_station.json'))
    id_to_traindists = load_json(train_path)

    trstation_nums = count_num(ids, id_to_traindists)
    df_data['train_station_num'] = trstation_nums

    has_trstations = num_to_has(trstation_nums)
    df_data['has_trstation'] = has_trstations

    # 公車站(數量+有無)
    bus_path = str(Path(distance_dir, 'bus_stop.json'))
    id_to_busdists = load_json(bus_path)

    busstop_nums = count_num(ids, id_to_busdists)
    df_data['bus_stop_num'] = busstop_nums

    has_busstops = num_to_has(busstop_nums)
    df_data['has_busstop'] = has_busstops
    df_data.head()

    # 捷運站(數量+有無)
    mrt_path = str(Path(distance_dir, 'mrt_station.json'))
    id_to_mrtdists = load_json(mrt_path)

    mrtstation_nums = count_num(ids, id_to_mrtdists)
    df_data['mrt_station_num'] = mrtstation_nums

    has_mrtstations = num_to_has(mrtstation_nums)
    df_data['has_mrtstation'] = has_mrtstations

    # 腳踏車站(數量+有無)
    bike_path = str(Path(distance_dir, 'bike.json'))
    id_to_bikedists = load_json(bike_path)

    bikestop_nums = count_num(ids, id_to_bikedists)
    df_data['bike_stop_num'] = bikestop_nums

    has_bikestops = num_to_has(bikestop_nums)
    df_data['has_bikestop'] = has_bikestops

    # 3. 計算教育機構外部資料距離
    # 國小
    elmschool_path = str(Path(distance_dir, 'elementary_school.json'))
    id_to_elmsdists = load_json(elmschool_path)

    elmschool_nums = count_num(ids, id_to_elmsdists)
    df_data['elem_school_num'] = elmschool_nums

    has_elmschools = num_to_has(elmschool_nums)
    df_data['has_elmschool'] = has_elmschools

    # 國中
    jh_school_path = str(Path(distance_dir,
                              'junior_high_school.json'))
    id_to_jhsdists = load_json(jh_school_path)

    jh_school_nums = count_num(ids, id_to_jhsdists)
    df_data['jh_school_num'] = jh_school_nums

    has_jhschools = num_to_has(jh_school_nums)
    df_data['has_jhschool'] = has_jhschools

    # 高中
    sh_school_path = str(Path(distance_dir,
                              'senior_high_school.json'))
    id_to_shsdists = load_json(sh_school_path)

    sh_school_nums = count_num(ids, id_to_shsdists)
    df_data['sh_school_num'] = sh_school_nums

    has_shschools = num_to_has(sh_school_nums)
    df_data['has_shschool'] = has_shschools

    # 大學
    college_path = str(Path(distance_dir, 'college.json'))
    id_to_coldists = load_json(college_path)

    college_nums = count_num(ids, id_to_coldists)
    df_data['college_num'] = college_nums

    has_colleges = num_to_has(college_nums)
    df_data['has_college'] = has_colleges

    # 4. 計算生活機能外部資料距離
    # 金融機構
    bank_path = str(Path(distance_dir, 'bank.json'))
    id_to_bankdists = load_json(bank_path)

    bank_nums = count_num(ids, id_to_bankdists)
    df_data['bank_num'] = bank_nums

    has_banks = num_to_has(bank_nums)
    df_data['has_bank'] = has_banks

    # ATM
    atm_path = str(Path(distance_dir, 'atm.json'))
    id_to_atmdists = load_json(atm_path)

    atm_nums = count_num(ids, id_to_atmdists)
    df_data['atm_num'] = atm_nums

    has_atms = num_to_has(atm_nums)
    df_data['has_atm'] = has_atms

    # 郵局
    post_path = str(Path(distance_dir, 'post.json'))
    id_to_postdists = load_json(post_path)

    post_nums = count_num(ids, id_to_postdists)
    df_data['post_num'] = post_nums

    has_posts = num_to_has(post_nums)
    df_data['has_post'] = has_posts

    # 便利商店
    cvs_path = str(Path(distance_dir, 'cvs.json'))
    id_to_cvsdists = load_json(cvs_path)

    cvs_nums = count_num(ids, id_to_cvsdists)
    df_data['cvs_num'] = cvs_nums

    has_cvses = num_to_has(cvs_nums)
    df_data['has_cvs'] = has_cvses

    # 5. 計算醫療院所外部資料距離
    # 西醫診所
    wsclinic_path = str(Path(distance_dir, 'western_clinic.json'))
    id_to_wscdists = load_json(wsclinic_path)

    wsclinic_nums = count_num(ids, id_to_wscdists)
    df_data['wsclinic_num'] = wsclinic_nums

    has_wsclinics = num_to_has(wsclinic_nums)
    df_data['has_wsclinic'] = has_wsclinics

    # 中醫診所
    chclinic_path = str(Path(distance_dir, 'chinese_clinic.json'))
    id_to_chcdists = load_json(chclinic_path)

    chclinic_nums = count_num(ids, id_to_chcdists)
    df_data['chclinic_num'] = chclinic_nums

    has_chclinics = num_to_has(chclinic_nums)
    df_data['has_chclinic'] = has_chclinics

    # 牙醫診所
    dtclinic_path = str(Path(distance_dir, 'dental_clinic.json'))
    id_to_dtcdists = load_json(dtclinic_path)

    dtclinic_nums = count_num(ids, id_to_dtcdists)
    df_data['dtclinic_num'] = dtclinic_nums

    has_dtclinics = num_to_has(dtclinic_nums)
    df_data['has_dtclinic'] = has_dtclinics

    # 衛生所
    healthctr_path = str(Path(distance_dir, 'health_center.json'))
    id_to_hctrdists = load_json(healthctr_path)

    healthctr_nums = count_num(ids, id_to_hctrdists)
    df_data['healthctr_num'] = healthctr_nums

    has_healthctrs = num_to_has(healthctr_nums)
    df_data['has_healthctstr(Path(distance_dir, '] = has_healthctrs

    # 醫院
    hospital_path = str(Path(distance_dir, 'hospital.json'))
    id_to_hospdists = load_json(hospital_path)

    hospital_nums = count_num(ids, id_to_hospdists)
    df_data['hospital_num'] = hospital_nums

    has_hospitals = num_to_has(hospital_nums)
    df_data['has_hospital'] = has_hospitals

    # 6. 修正分佈不均的資料
    num_cols = [
        'train_station_num', 'bus_stop_num', 'mrt_station_num',
        'bike_stop_num', 'elem_school_num', 'jh_school_num', 'sh_school_num',
        'college_num', 'bank_num', 'atm_num', 'post_num', 'cvs_num',
        'wsclinic_num', 'chclinic_num', 'dtclinic_num', 'healthctr_num',
        'hospital_num'
    ]

    # 檢查傾斜程度
    skewnesses = list()
    for col in num_cols:
        skewness = skew(df_data[col])
        skewnesses.append(skewness)

    df_skewness = pd.DataFrame({'col': num_cols, 'skewness': skewnesses})
    df_skewness.sort_values(by=['skewness'], ascending=False, inplace=True)

    # 進行 Box-Cox Transform
    lam = 0.15
    transform_cols = list()
    for col in num_cols:
        df_data[f'{col}_transform'] = boxcox1p(df_data[col], lam)
        transform_cols.append(f'{col}_transform')

    # # 7. 篩選需要的欄位
    # use_cols = [
    #     'ID', '縣市', '鄉鎮市區', '路名', '經度', '緯度', '縣市_label', '縣市鄉鎮市區_label',
    #     '經度_transformed', '緯度_transformed', '土地面積_transformed',
    #     '總建物面積_transformed', '移轉層次_cat', '總樓層數_cat', '屋齡_cat', '主要用途_label',
    #     '主要建材_label', '建物型態_label', 'has_parking', '車位面積_transformed',
    #     'cd_rprice', 'cdr_rprice', 'train_station_num_transform',
    #     'bus_stop_num_transform', 'mrt_station_num_transform',
    #     'bike_stop_num_transform', 'elem_school_num_transform',
    #     'jh_school_num_transform', 'sh_school_num_transform',
    #     'college_num_transform', 'bank_num_transform', 'atm_num_transform',
    #     'post_num_transform', 'cvs_num_transform', 'wsclinic_num_transform',
    #     'chclinic_num_transform', 'dtclinic_num_transform',
    #     'healthctr_num_transform', 'hospital_num_transform'
    # ]
    # df_data_merged = df_data.loc[:, use_cols]

    return df_data
