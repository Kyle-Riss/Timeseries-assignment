import pandas as pd

filtered_ads_data_file = 'SGO_ADS_CA_2024_2025.csv'

try:
    df_ads_final = pd.read_csv(filtered_ads_data_file)
    print(f"'{filtered_ads_data_file}' 파일을 성공적으로 로드했습니다.")
    print(f"로드된 ADS 데이터 건수: {len(df_ads_final)}")
    print("\n--- 로드된 ADS 데이터 처음 5개 행 ---")
    print(df_ads_final.head())
    print("\n--- 로드된 ADS 데이터 컬럼 정보 ---")
    df_ads_final.info()
except FileNotFoundError:
    print(f"오류: '{filtered_ads_data_file}' 파일을 찾을 수 없습니다. 파일 경로와 이름을 확인해주세요.")
    exit()
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")
    exit()

vin_column_name_in_filtered_data = 'VIN'
vin_unknown_column_name_in_filtered_data = 'VIN - Unknown'

if vin_column_name_in_filtered_data in df_ads_final.columns:
    print(f"\n--- '{vin_column_name_in_filtered_data}' 컬럼 (처음 10개) ---")
    print(df_ads_final[vin_column_name_in_filtered_data].head(10))
else:
    print(f"경고: '{vin_column_name_in_filtered_data}' 컬럼을 찾을 수 없습니다.")

if vin_unknown_column_name_in_filtered_data in df_ads_final.columns:
    print(f"\n--- '{vin_unknown_column_name_in_filtered_data}' 컬럼 고유값 및 개수 ---")
    print(df_ads_final[vin_unknown_column_name_in_filtered_data].value_counts(dropna=False))
else:
    print(f"경고: '{vin_unknown_column_name_in_filtered_data}' 컬럼을 찾을 수 없습니다.")