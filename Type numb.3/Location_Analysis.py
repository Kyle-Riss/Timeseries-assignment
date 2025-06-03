import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf' 
font_name = fm.FontProperties(fname=font_path, size=10).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

geocoded_ads_file = 'SGO_ADS_CA_2024_2025_with_Geocodes.csv'
ads_city_counts = pd.DataFrame()
try:
    df_ads_geo = pd.read_csv(geocoded_ads_file)
    print(f"\n'{geocoded_ads_file}' 파일을 성공적으로 로드했습니다. (총 {len(df_ads_geo)} 건의 ADS 사고 데이터)")
    if 'City' in df_ads_geo.columns:
        ads_city_counts = df_ads_geo['City'].value_counts().reset_index()
        ads_city_counts.columns = ['City', 'ADS_Accident_Count']
        print("\n--- ADS 사고 발생 건수 (상위 10개 도시) ---")
        print(ads_city_counts.head(10))
    else:
        print(f"'City' 컬럼을 '{geocoded_ads_file}' 파일에서 찾을 수 없습니다.")
except FileNotFoundError:
    print(f"오류: '{geocoded_ads_file}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"'{geocoded_ads_file}' 파일 처리 중 오류 발생: {e}")

geocoded_adas_file = 'SGO_ADAS_2024_2025_CA_with_Geocodes.csv'
adas_city_counts = pd.DataFrame()
try:
    df_adas_geo = pd.read_csv(geocoded_adas_file)
    print(f"\n'{geocoded_adas_file}' 파일을 성공적으로 로드했습니다. (총 {len(df_adas_geo)} 건의 ADAS 사고 데이터)")
    if 'City' in df_adas_geo.columns:
        adas_city_counts = df_adas_geo['City'].value_counts().reset_index()
        adas_city_counts.columns = ['City', 'ADAS_Accident_Count']
        print("\n--- ADAS 사고 발생 건수 (상위 10개 도시) ---")
        print(adas_city_counts.head(10))
    else:
        print(f"'City' 컬럼을 '{geocoded_adas_file}' 파일에서 찾을 수 없습니다.")
except FileNotFoundError:
    print(f"오류: '{geocoded_adas_file}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"'{geocoded_adas_file}' 파일 처리 중 오류 발생: {e}")

if not ads_city_counts.empty and not adas_city_counts.empty:
    combined_city_counts = pd.merge(ads_city_counts, adas_city_counts, on='City', how='outer')
    combined_city_counts = combined_city_counts.fillna(0)
    combined_city_counts['Total_Accidents'] = combined_city_counts['ADS_Accident_Count'] + combined_city_counts['ADAS_Accident_Count']
    combined_city_counts = combined_city_counts.sort_values(by='Total_Accidents', ascending=False)
    print("\n--- ADS 및 ADAS 통합 도시별 사고 건수 (상위 10개 도시) ---")
    print(combined_city_counts.head(10))

    top_n = 10
    plot_data = combined_city_counts.head(top_n).set_index('City')

    if not plot_data.empty:
        plot_data[['ADS_Accident_Count', 'ADAS_Accident_Count']].plot(
            kind='bar',
            figsize=(15, 8),
            width=0.8
        )
        plt.title(f'캘리포니아 주 상위 {top_n}개 도시별 ADS/ADAS 사고 건수 (2024-2025)', fontsize=16)
        plt.xlabel('도시 (City)', fontsize=12)
        plt.ylabel('사고 건수 (Accident Count)', fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='시스템 유형', fontsize=10)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()

        graph_output_filename = 'city_accident_counts_comparison.png'
        try:
            plt.savefig(graph_output_filename)
            print(f"\n그래프가 '{graph_output_filename}' 파일로 성공적으로 저장되었습니다.")
            print(f"저장된 파일을 확인해보세요!")
        except Exception as e:
            print(f"\n오류: 그래프 저장 중 문제가 발생했습니다 - {e}")
    else:
        print("\n그래프를 그릴 데이터가 없습니다 (필터링된 도시가 없음).")
else:
    print("\nADS 또는 ADAS 도시별 사고 건수 데이터가 준비되지 않아 그래프 생성을 진행할 수 없습니다.")