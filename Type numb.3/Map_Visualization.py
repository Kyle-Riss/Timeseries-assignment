import pandas as pd
import folium

geocoded_adas_file = 'SGO_ADAS_2024_2025_CA_with_Geocodes.csv'

try:
    df_adas_geocoded = pd.read_csv(geocoded_adas_file)
    print(f"'{geocoded_adas_file}' 파일을 성공적으로 로드했습니다. (총 {len(df_adas_geocoded)} 건)")
except FileNotFoundError:
    print(f"오류: '{geocoded_adas_file}' 파일을 찾을 수 없습니다. 파일 경로와 이름을 확인해주세요.")
    print("이 파일은 ADAS 데이터에 대한 Geocoding 스크립트 실행 후 생성되어야 합니다.")
    exit()
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")
    exit()

df_map_data = df_adas_geocoded.dropna(subset=['Approx_Latitude', 'Approx_Longitude']).copy()
print(f"지도에 표시할 유효한 ADAS 좌표 데이터 건수: {len(df_map_data)}")

if df_map_data.empty:
    print("지도에 표시할 ADAS 데이터가 없습니다.")
    exit()

map_center_lat = 36.7783
map_center_lon = -119.4179
initial_zoom = 6

california_map_adas = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=initial_zoom)

for index, row in df_map_data.iterrows():
    city_name = row.get('City', 'N/A')
    
    folium.CircleMarker(
        location=[row['Approx_Latitude'], row['Approx_Longitude']],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        tooltip=f"City: {city_name} (ADAS)<br>Lat: {row['Approx_Latitude']:.4f}<br>Lon: {row['Approx_Longitude']:.4f}"
    ).add_to(california_map_adas)

print("\n--- ADAS 데이터 지도 생성 중 ---")

map_output_filename_adas = 'adas_ca_accidents_map.html'
try:
    california_map_adas.save(map_output_filename_adas)
    print(f"\nADAS 지도가 '{map_output_filename_adas}' 파일로 성공적으로 저장되었습니다.")
    print(f"저장된 파일을 웹 브라우저로 열어보세요!")
except Exception as e:
    print(f"\n오류: 지도 저장 중 문제가 발생했습니다 - {e}")