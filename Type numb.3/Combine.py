import pandas as pd
import numpy as np

ads_file = 'ads_data.csv'
adas_file = 'adas_data.csv'

try:
    df_ads = pd.read_csv(ads_file)
    df_adas = pd.read_csv(adas_file)

    print(f"원본 ADS 데이터 형태: {df_ads.shape}")
    print(f"원본 ADAS 데이터 형태: {df_adas.shape}")

    df_ads['SystemType'] = 'ADS'
    df_adas['SystemType'] = 'ADAS'

    combined_df = pd.concat([df_ads, df_adas], ignore_index=True)

    print(f"통합된 데이터 형태: {combined_df.shape}")
    print("\n'SystemType'이 추가된 통합 DataFrame의 상위 5행:")
    print(combined_df[['VIN', 'Make', 'Model', 'Automation System Engaged?', 'SystemType', 'Incident Date']].head())
    print("\n'SystemType'이 추가된 통합 DataFrame의 하위 5행:")
    print(combined_df[['VIN', 'Make', 'Model', 'Automation System Engaged?', 'SystemType', 'Incident Date']].tail())

    combined_df['Incident Date_str'] = combined_df['Incident Date'].fillna('')
    combined_df['Incident Time_str'] = combined_df['Incident Time (24:00)'].fillna('00:00')

    combined_df['Incident_Datetime_Temp'] = combined_df['Incident Date_str'] + ' ' + combined_df['Incident Time_str']

    combined_df['Incident_Datetime'] = pd.to_datetime(combined_df['Incident_Datetime_Temp'], errors='coerce')

    combined_df.drop(columns=['Incident Date_str', 'Incident Time_str', 'Incident_Datetime_Temp'], inplace=True)

    print(f"\nNaT (datetime 변환 실패)를 가진 행의 수: {combined_df['Incident_Datetime'].isna().sum()}")

    combined_df.sort_values(by='Incident_Datetime', inplace=True)

    print("\nIncident_Datetime으로 정렬된 통합 DataFrame (상위 5행):")
    print(combined_df[['Incident_Datetime', 'SystemType', 'Make', 'Model']].head())
    print("\nIncident_Datetime으로 정렬된 통합 DataFrame (하위 5행):")
    print(combined_df[['Incident_Datetime', 'SystemType', 'Make', 'Model']].tail())

    combined_df.to_csv('combined_ads_adas_data.csv', index=False)
    print("\n통합된 데이터가 'combined_ads_adas_data.csv'로 저장되었습니다.")

except FileNotFoundError:
    print("오류: 'ads_data.csv'와 'adas_data.csv' 파일이 스크립트와 같은 디렉토리에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")