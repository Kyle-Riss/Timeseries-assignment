import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

combined_file = 'combined_ads_adas_data.csv'
try:
    df = pd.read_csv(combined_file)
    print(f"'{combined_file}' 파일 로드 성공. 데이터 형태: {df.shape}")
except FileNotFoundError:
    print(f"오류: '{combined_file}' 파일을 찾을 수 없습니다. 먼저 통합 스크립트를 실행했는지 확인해주세요.")
    exit()

df['Incident_Datetime'] = pd.to_datetime(df['Incident_Datetime'], errors='coerce')
if df['Incident_Datetime'].isna().sum() > 0:
    print(f"경고: Incident_Datetime 변환 중 NaT 발생. NaT 개수: {df['Incident_Datetime'].isna().sum()}")
    df.dropna(subset=['Incident_Datetime'], inplace=True)
    print(f"NaT 행 삭제 후 데이터 형태: {df.shape}")

df.sort_values(by='Incident_Datetime', inplace=True)

print("\n--- 1. 데이터 클리닝 및 결측치 처리 ---")

version_cols = [
    'ADAS/ADS System Version', 'ADAS/ADS System Version CBI',
    'ADAS/ADS Hardware Version', 'ADAS/ADS Hardware Version CBI',
    'ADAS/ADS Software Version', 'ADAS/ADS Software Version CBI'
]

for col in version_cols:
    if col in df.columns:
        df[col] = df[col].replace(
            '[REDACTED, MAY CONTAIN CONFIDENTIAL BUSINESS INFORMATION]', 'REDACTED'
        )
        df[col].fillna('UNKNOWN', inplace=True)
        print(f"'{col}' 컬럼의 [REDACTED] 및 NaN 값 처리 완료.")
    else:
        print(f"경고: '{col}' 컬럼이 데이터에 없습니다.")

if 'Mileage' in df.columns:
    if df['Mileage'].isna().sum() > 0:
        median_mileage = df['Mileage'].median()
        df['Mileage'].fillna(median_mileage, inplace=True)
        print(f"'Mileage' 컬럼 NaN 값 중앙값({median_mileage})으로 대체 완료.")

if 'Posted Speed Limit (MPH)' in df.columns:
    if df['Posted Speed Limit (MPH)'].isna().sum() > 0:
        median_speed_limit = df['Posted Speed Limit (MPH)'].median()
        df['Posted Speed Limit (MPH)'].fillna(median_speed_limit, inplace=True)
        print(f"'Posted Speed Limit (MPH)' 컬럼 NaN 값 중앙값({median_speed_limit})으로 대체 완료.")

print("\n--- 2. 범주형 변수 인코딩 (One-Hot Encoding) ---")

categorical_cols = [
    'SystemType', 'Make', 'Model', 'Driver / Operator Type', 'Roadway Type',
    'Roadway Surface', 'Lighting', 'Crash With', 'Highest Injury Severity Alleged',
    'ADAS/ADS System Version', 'ADAS/ADS Hardware Version', 'ADAS/ADS Software Version',
    'Weather - Clear', 'Weather - Snow', 'Weather - Cloudy', 'Weather - Fog/Smoke',
    'Weather - Rain', 'Weather - Severe Wind', 'Weather - Other'
]

categorical_cols_exist = [col for col in categorical_cols if col in df.columns]

df_encoded = pd.get_dummies(df, columns=categorical_cols_exist, dummy_na=False)
print(f"범주형 변수 인코딩 완료. 새로운 데이터 형태: {df_encoded.shape}")
print(f"인코딩된 컬럼의 예시 (상위 5개):\n{df_encoded.filter(like='SystemType_').head()}")

print("\n--- 3. 수치형 변수 스케일링 (Min-Max Scaling) ---")

numerical_cols = [
    'Mileage', 'Posted Speed Limit (MPH)', 'Latitude', 'Longitude',
    'SV Precrash Speed (MPH)', 'Report Version'
]

numerical_cols_exist = []
for col in numerical_cols:
    if col in df_encoded.columns and pd.api.types.is_numeric_dtype(df_encoded[col]):
        if df_encoded[col].isna().sum() > 0:
            print(f"경고: 수치형 컬럼 '{col}'에 NaN 값이 있습니다. 스케일링 전에 처리해주세요.")
        else:
            numerical_cols_exist.append(col)
    else:
        print(f"경고: '{col}' 컬럼이 존재하지 않거나 수치형이 아닙니다.")

if numerical_cols_exist:
    scaler = MinMaxScaler()
    df_encoded[numerical_cols_exist] = scaler.fit_transform(df_encoded[numerical_cols_exist])
    print(f"수치형 변수 스케일링 완료 (Min-Max Scaling).")
    print(f"스케일링된 컬럼의 예시 (상위 5개):\n{df_encoded[numerical_cols_exist].head()}")
else:
    print("스케일링할 수치형 컬럼이 없습니다.")

print("\n--- 전처리 요약 ---")
print(f"최종 전처리된 데이터 형태: {df_encoded.shape}")
print("전처리된 데이터의 일부:\n", df_encoded[['Incident_Datetime', 'SystemType_ADS', 'SystemType_ADAS', 'Mileage', 'Posted Speed Limit (MPH)']].head())

df_encoded.to_csv('preprocessed_combined_data.csv', index=False)
print("\n전처리된 데이터가 'preprocessed_combined_data.csv'로 저장되었습니다.")