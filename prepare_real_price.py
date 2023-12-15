import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple, Set
from util.csv_util import save_csv


def extract_from_raw(data_path: str, city: str) -> pd.DataFrame:

    # 1. 讀取不動產買賣資料
    df_realestate = pd.read_excel(data_path, sheet_name='不動產買賣')

    # 2. 過濾交易標的，只挑「房地(土地+建物)」、「房地(土地+建物)+車位」的資料
    transaction_signs = ('房地(土地+建物)', '房地(土地+建物)+車位')
    df_realestate = df_realestate[df_realestate['交易標的'].isin(
        transaction_signs)]

    # 3. 篩選欄位
    use_cols = [
        '交易年月日', '編號', '鄉鎮市區', '交易標的', '土地位置建物門牌', '移轉層次', '總樓層數', '建物型態',
        '主要用途', '主要建材', '總價元', '單價元平方公尺', '備註'
    ]
    df_realestate = df_realestate.loc[:, use_cols]

    # 4. 計算縣市欄位
    df_realestate['縣市'] = city

    # 5. 移除重複資料
    df_realestate.drop_duplicates(inplace=True)

    # 6. 處理建物資料
    df_building = pd.read_excel(data_path, sheet_name='建物')

    # 7. 篩選欄位
    df_building = df_building.loc[:, ['編號', '屋齡']]

    # 8. 移除重複資料
    df_building.drop_duplicates(inplace=True)

    # 9. 合併資料
    df_merge = pd.merge(df_realestate, df_building, on='編號')

    # 10. 移除重複資料
    df_merge.drop_duplicates(inplace=True)

    return df_merge


def read_and_extract(rp_dir: str) -> pd.DataFrame:

    # 身分證字號英文與縣市對照表
    capital_to_city = {
        'A': '臺北市',
        'B': '臺中市',
        'C': '基隆市',
        'D': '臺南市',
        'E': '高雄市',
        'F': '新北市',
        'G': '宜蘭縣',
        'H': '桃園市',
        'J': '新竹縣',
        'K': '苗栗縣',
        'L': '臺中縣',
        'M': '南投縣',
        'N': '彰化縣',
        'P': '雲林縣',
        'Q': '嘉義縣',
        'R': '臺南縣',
        'S': '高雄縣',
        'T': '屏東縣',
        'U': '花蓮縣',
        'V': '臺東縣',
        'X': '澎湖縣',
        'Y': '陽明山',
        'W': '金門縣',
        'Z': '連江縣',
        'I': '嘉義市',
        'O': '新竹市'
    }

    # 2. 讀取實價登錄資料夾裡面的檔案
    df = pd.DataFrame({})
    for csv_path in Path(rp_dir).rglob('*.xls'):

        # 檢查檔名結尾，只用不動產的資料('_lvr_land_a.xls')
        if not csv_path.stem.endswith('_lvr_land_a'):
            continue

        # 計算縣市名稱
        prefix = csv_path.stem.split('_')[0].upper()
        city = capital_to_city.get(prefix)

        # 處理資料
        try:
            _df = extract_from_raw(str(csv_path), city)
            df = pd.concat([df, _df])
        except Exception as e:
            print(e)

    return df


def fix_tai_prefix(text) -> str:

    if not text.startswith('臺'):
        return text

    new_text = f'台{text[1:]}'

    return new_text


def load_ris() -> Set[str]:

    # 1. 載入資料
    li_path = r'ris_data/村里代碼檔-111年9月（UTF8）.txt'
    li_texts = None
    with open(li_path, 'r', encoding='utf-8') as f:
        li_texts = f.read().split('\n')

    # 2. 整理村里名稱
    lis = set()
    for _ in li_texts:
        li = _.split(',')[-1]

        if _ == '':
            continue

        lis.add(li)

    return lis


def map_variant(text: str) -> str:

    word_map = {
        '東': '東',
        '巿': '市',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '中壢市內壢里三鄰成功路': '成功路'
    }

    for old, new in word_map.items():
        text = text.replace(old, new)

    return text


def split_road_name(df: pd.DataFrame, lis: set) -> pd.DataFrame:

    # 1. 處理路名
    roads: List[str] = list()
    for row in df.itertuples():

        road = row.土地位置建物門牌
        city = row.縣市
        depart = row.鄉鎮市區

        # 移除縣市
        if city in road:
            road = road.replace(city, '')

        # 移除鄉鎮市區(小台)
        if depart in road:
            road = road.replace(depart, '')

        # 移除鄉鎮市區(大台)
        depart_big = depart.replace('台', '臺')
        if depart_big in road:
            road = road.replace(depart_big, '')

        # 從路名後的數字切割
        number_matches = [_ for _ in re.finditer(r'[０１２３４５６７８９]+', road)]
        if number_matches:
            road = road[:number_matches[0].start()]

        # 移除巷村
        # X路Y段Z巷/村
        alley_match1 = re.search(r'(.{1,10}路.{1,10}段)(.{1,10}[巷村])', road)
        if alley_match1:
            road = alley_match1.group(1)

        # X街Y巷/村
        alley_match2 = re.search(r'(.{1,10}[路街])(.{1,10}[巷村])', road)
        if alley_match2:
            road = alley_match2.group(1)

        for li in lis:
            if road.startswith(li):
                road = road.replace(li, '')
                break

        roads.append(road)

    # 2. 設定路名
    df['路名'] = roads

    return df


def convert_level(text) -> int:

    chs = '一二三四五六七八九十'
    nums = [_ for _ in range(1, 11)]
    ch_to_number = {ch: num for ch, num in zip(chs, nums)}

    # 1. 判斷例外情況
    if text in ('(空白)', '00Z', '00Y', '地下層', '000', '見使用執照', '見其他登記事項', '099',
                '地下二層', '0'):
        return 0

    # 2. 移除「層」字
    if text.endswith('層'):
        text = text[:-1]

    total = 0
    for i, ch in enumerate(text):
        if ch == '十' and i == 0:
            total = 1

        ch_val = ch_to_number.get(ch)
        if ch == '十':
            total = total * 10
        else:
            total = total + ch_val

    return total


def calculate_mean(
        df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # 1. 計算各縣市平均價格
    city_mean = df.groupby('縣市')['單價元坪'].mean()

    df_city_mean = pd.DataFrame({
        'city': city_mean.index,
        'mean': city_mean.values
    })

    # 2. 計算各縣市、鄉鎮市區平均價格
    cd_mean = df.groupby(['縣市', '鄉鎮市區'])['單價元坪'].mean()
    citys = [_[0] for _ in cd_mean.index]
    departs = [_[1] for _ in cd_mean.index]
    means = cd_mean.values

    df_cdmean = pd.DataFrame({'city': citys, 'depart': departs, 'mean': means})

    # 3. 計算各縣市、鄉鎮市區、路名平均價格
    cdr_mean = df.groupby(['縣市', '鄉鎮市區', '路名', '總樓層數', '屋齡'])['單價元坪'].mean()

    citys = [_[0] for _ in cdr_mean.index]
    departs = [_[1] for _ in cdr_mean.index]
    roads = [_[2] for _ in cdr_mean.index]
    levels = [_[3] for _ in cdr_mean.index]
    ages = [_[4] for _ in cdr_mean.index]
    means = cdr_mean.values

    df_cdrmean = pd.DataFrame({
        'city': citys,
        'depart': departs,
        'road': roads,
        'level': levels,
        'age': ages,
        'mean': means
    })

    return df_city_mean, df_cdmean, df_cdrmean


def prepare(rp_dir: str) -> None:

    # 1. 讀取資料並整理
    df_price = read_and_extract(rp_dir)

    # 2. 移除備註欄位中有「親友」字樣的資料
    # NOTE: 有包含親友字樣的資料的成交價格與實際狀況落差較大，所以捨棄不用
    remark_counts = df_price['備註'].value_counts()
    special_remarks = [_ for _ in remark_counts.index if '親友' in _]
    df_price = df_price[~df_price['備註'].isin(special_remarks)]

    # 3. 統一用字，大台轉小台
    df_price['土地位置建物門牌'] = df_price['土地位置建物門牌'].apply(fix_tai_prefix)
    df_price['縣市'] = df_price['縣市'].apply(fix_tai_prefix)

    # 4. 修正地址中的異體字
    df_price['土地位置建物門牌'] = df_price['土地位置建物門牌'].apply(map_variant)

    # 5. 分割路名
    lis = load_ris()
    df_price = split_road_name(df_price, lis)

    # 6. 平方公尺轉換成坪
    df_price['單價元坪'] = df_price['單價元平方公尺'] * 3.30579

    # 7. 轉換總樓層數
    df_price['總樓層數'].fillna('0', inplace=True)
    df_price['總樓層數'] = df_price['總樓層數'].apply(convert_level)

    # 8. 移除重複資料
    df_price.drop_duplicates(inplace=True)

    # 9. 計算平均價格
    df_city_mean, df_cdmean, df_cdrmean = calculate_mean(df_price)

    # 10. 存擋
    save_csv(df_city_mean, r'real_price/cmean.csv')
    save_csv(df_cdmean, r'real_price/cdmean.csv')
    save_csv(df_cdrmean, r'real_price/cdrmean.csv')
    save_csv(df_price, r'real_price/real_price.csv')


if __name__ == '__main__':

    rp_dir = r'real_price/'
    prepare(rp_dir)
