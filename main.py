import eda_data
import calculate_distance
import merge_external
import merge_real_price
import train_model
import preprocess_data
import predict_data
import pandas as pd
from util.csv_util import save_csv


def prepare_data(df: pd.DataFrame,
                 data_type: str,
                 external_dir: str,
                 is_train: bool = True) -> pd.DataFrame:

    # 1. 資料清洗 + EDA
    if is_train:
        df = eda_data.clean(df)
    else:
        df = preprocess_data.preprocess(df)

    # 2. 計算外部資料
    external_dir = r'official_dataset/30_Training Dataset_V2/external_data'
    calculate_distance.calculate(df, data_type, external_dir)

    # 3. 合併外部資料
    distance_dir = f'distance_{data_type}/'
    df_mg = merge_external.merge_ext(df, distance_dir)

    # 4. 合併實價登錄資料
    df_mg = merge_real_price.merge_rp(df_mg)

    return df_mg


def run():

    # 1. 準備訓練資料
    train_path = 'official_dataset/30_Training Dataset_V2/training_data.csv'
    df_train = pd.read_csv(train_path)

    # 2. 處理訓練資料
    external_dir = r'official_dataset/30_Training Dataset_V2/external_data'
    df_train_mg = prepare_data(df_train, 'train', external_dir, is_train=True)

    # 3. 存檔
    save_csv(df_train_mg, r'dataset/train_merged.csv')

    # 4. 訓練模型
    train_model.train(df_train_mg)

    # 5. 準備預測資料
    test_path = 'official_dataset/Testing_Dataset_V2/public_dataset.csv'
    df_test = pd.read_csv(test_path)

    # 6. 處理預測資料
    external_dir = r'official_dataset/30_Training Dataset_V2/external_data'
    df_test_mg = prepare_data(df_test, 'public', external_dir, is_train=False)

    # 7. 存檔
    save_csv(df_test_mg, r'dataset/test_merged.csv')

    # 8. 預測資料
    predict_data.predict(df_test_mg)


if __name__ == '__main__':

    run()
