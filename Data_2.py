import pandas as pd

adas_csv_file = 'SGO-2021-01_Incident_Reports_ADAS.csv'
try:
    df_adas = pd.read_csv(adas_csv_file)
    print(f"'{adas_csv_file}' 파일을 성공적으로 로드했습니다.")
    print("--- 원본 데이터프레임 정보 (ADAS.csv) ---")
    df_adas.info()
    print("\n--- 원본 데이터 처음 5개 행 (ADAS.csv) ---")
    print(df_adas.head())
except FileNotFoundError:
    print(f"오류: '{adas_csv_file}' 파일을 찾을 수 없습니다. 파일 경로와 이름을 확인해주세요.")
    exit()
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")
    exit()

date_column_actual_name = 'Incident Date'
state_column_actual_name = 'State'
california_value_in_data = 'CA'

if date_column_actual_name not in df_adas.columns:
    print(f"\n오류: 날짜 컬럼 '{date_column_actual_name}'을 찾을 수 없습니다. 컬럼명을 확인해주세요.")
    exit()
try:
    df_adas[date_column_actual_name] = pd.to_datetime(df_adas[date_column_actual_name], errors='coerce')
    df_adas_filtered = df_adas.dropna(subset=[date_column_actual_name]).copy()

    df_adas_filtered = df_adas_filtered[
        (df_adas_filtered[date_column_actual_name].dt.year >= 2024) &
        (df_adas_filtered[date_column_actual_name].dt.year <= 2025)
    ]
    print(f"\n--- 2024-2025년 데이터 필터링 후 {len(df_adas_filtered)} 건 ---")
except Exception as e:
    print(f"\n오류: 날짜 컬럼 '{date_column_actual_name}' 처리 중 오류 발생: {e}")
    exit()

if state_column_actual_name not in df_adas_filtered.columns:
    print(f"\n오류: 주(State) 컬럼 '{state_column_actual_name}'을 찾을 수 없습니다. 컬럼명을 확인해주세요.")
    df_final_filtered_adas = df_adas_filtered
else:
    df_final_filtered_adas = df_adas_filtered[
        df_adas_filtered[state_column_actual_name].astype(str).str.strip().str.upper() == california_value_in_data.strip().upper()
    ].copy()
    print(f"\n--- 캘리포니아 주 데이터 필터링 후 {len(df_final_filtered_adas)} 건 ---")

if not df_final_filtered_adas.empty:
    print("\n--- 최종 필터링된 데이터 (ADAS, 2024-2025년, 캘리포니아) ---")
    print(df_final_filtered_adas.head())
    print(f"\n최종적으로 {len(df_final_filtered_adas)} 건의 데이터가 선택되었습니다.")

    df_final_filtered_adas.to_csv('SGO_ADAS_2024_2025_CA.csv', index=False)
    print("필터링된 ADAS 데이터가 'SGO_ADAS_2024_2025_CA.csv' 파일로 저장되었습니다.")
else:
    print("\n해당 조건에 맞는 ADAS 데이터가 없습니다.")