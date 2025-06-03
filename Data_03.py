import pandas as pd

current_csv_file = 'SGO-2021-01_Incident_Reports_OTHER.csv'
try:
    df_current = pd.read_csv(current_csv_file)
    print(f"'{current_csv_file}' 파일을 성공적으로 로드했습니다.")
    print(f"--- 원본 데이터프레임 정보 ({current_csv_file}) ---")
    df_current.info()
    print(f"\n--- 원본 데이터 처음 5개 행 ({current_csv_file}) ---")
    print(df_current.head())
except FileNotFoundError:
    print(f"오류: '{current_csv_file}' 파일을 찾을 수 없습니다. 파일 경로와 이름을 확인해주세요.")
    exit()
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")
    exit()

date_column_actual_name = 'Incident Date'
state_column_actual_name = 'State'
california_value_in_data = 'CA '

if date_column_actual_name not in df_current.columns:
    print(f"\n오류: 날짜 컬럼 '{date_column_actual_name}'을 찾을 수 없습니다. 컬럼명을 확인해주세요.")
    exit()
try:
    df_current[date_column_actual_name] = pd.to_datetime(df_current[date_column_actual_name], errors='coerce')
    df_current_filtered_by_date = df_current.dropna(subset=[date_column_actual_name]).copy()

    df_current_filtered_by_date = df_current_filtered_by_date[
        (df_current_filtered_by_date[date_column_actual_name].dt.year >= 2024) &
        (df_current_filtered_by_date[date_column_actual_name].dt.year <= 2025)
    ]
    print(f"\n--- 2024-2025년 데이터 필터링 후 {len(df_current_filtered_by_date)} 건 ---")
except Exception as e:
    print(f"\n오류: 날짜 컬럼 '{date_column_actual_name}' 처리 중 오류 발생: {e}")
    exit()

if state_column_actual_name not in df_current_filtered_by_date.columns:
    print(f"\n오류: 주(State) 컬럼 '{state_column_actual_name}'을 찾을 수 없습니다. 컬럼명을 확인해주세요.")
    df_final_filtered = df_current_filtered_by_date
else:
    print(f"\n--- 날짜 필터링 후 남은 데이터의 '{state_column_actual_name}' 컬럼 고유값 ({current_csv_file}) ---")
    print(df_current_filtered_by_date[state_column_actual_name].unique())

    df_final_filtered = df_current_filtered_by_date[
        df_current_filtered_by_date[state_column_actual_name].astype(str).str.strip().str.upper() == california_value_in_data.strip().upper()
    ].copy()
    print(f"\n--- 캘리포니아 주 데이터 필터링 후 {len(df_final_filtered)} 건 ---")

if not df_final_filtered.empty:
    print(f"\n--- 최종 필터링된 데이터 ({current_csv_file}, 2024-2025년, 캘리포니아) ---")
    print(df_final_filtered.head())
    print(f"\n최종적으로 {len(df_final_filtered)} 건의 데이터가 선택되었습니다.")

    # Uncomment the lines below to save the filtered data
    # df_final_filtered.to_csv('SGO_OTHER_2024_2025_CA.csv', index=False)
    # print("필터링된 OTHER 데이터가 'SGO_OTHER_2024_2025_CA.csv' 파일로 저장되었습니다.")
else:
    print(f"\n해당 조건에 맞는 데이터가 {current_csv_file} 파일에 없습니다.")