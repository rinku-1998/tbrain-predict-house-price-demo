# tbrain-predict-house-price-demo
TBrain 永豐AI GO競賽-攻房戰，第 74 名 (別人恐懼我貪婪) 的解決方案。

## 環境
程式在以下環境測試得以正常執行：
- Python 3.11.2
- Windows 11
- macOS 14.2

## 快速指南

1. 安裝套件

```shell
pip install -r requirements.txt
```

2. 將官方提供的「訓練集」與「公開測試集」放到 `official_dataset/` 資料夾內，放完後目錄如下：

```
official_dataset
├── 30_Training Dataset_V2
│   ├── external_data
│   │   ├── ATM資料.csv
│   │   ├── 便利商店.csv
│   │   ├── 公車站點資料.csv
│   │   ├── 國中基本資料.csv
│   │   ├── 國小基本資料.csv
│   │   ├── 大學基本資料.csv
│   │   ├── 捷運站點資料.csv
│   │   ├── 火車站點資料.csv
│   │   ├── 郵局據點資料.csv
│   │   ├── 高中基本資料.csv
│   │   ├── 腳踏車站點資料.csv
│   │   ├── 醫療機構基本資料.csv
│   │   └── 金融機構基本資料.csv
│   └── training_data.csv
└── Testing_Dataset_V2
    ├── public_dataset.csv
    └── public_submission_template.csv
```

3. 放實價登錄資料到 `real_price/` 內，放完後檔案結構如下，每個季度的資料夾命名應為 `lvr_landxls_111_Q1`，且其底下應有不同的 `.xls` 檔案：

```
real_price
├── lvr_landxls_111_Q1
├── a_lvr_land_a.xls
│   ├── a_lvr_land_b.xls
│   ├── a_lvr_land_c.xls
│   ├── b_lvr_land_a.xls
│   ├── b_lvr_land_b.xls
│   ├── b_lvr_land_c.xls
├── lvr_landxls_111_Q2
├── lvr_landxls_111_Q3
├── lvr_landxls_111_Q4
├── lvr_landxls_112_Q1
├── lvr_landxls_112_Q2
├── lvr_landxls_112_Q3
```

4. 一鍵執行資料處理、訓練與預測，各個流程輸出的資料夾如下：

- 資料處理：`dataset/`
- 訓練模型權重：`output/model/`
- 預測結果：`output/submissions`

  5.各個階段所需要的檔案，如編碼值、平移值、實價登錄價格等如下：

- 實價登錄平均價格參照資料：`real_price/cdr.json`
- 對數變換（Log Transform）平移數值：`offset/col_to_offset.json`
- 類別型資料編碼值：`encode/*.json`
- 外部機能設施距離：`distance_[train/public]`/\*.json

## 其他
1. 因比賽政策，故無法提供官方提供的資料集。