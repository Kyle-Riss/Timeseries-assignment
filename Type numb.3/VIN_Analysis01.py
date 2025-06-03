import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

current_csv_file = 'SGO_ADAS_2024_2025_CA.csv'

try:
    df_to_geocode = pd.read_csv(current_csv_file)
    print(f"'{current_csv_file}' 파일을 성공적으로 로드했습니다. (총 {len(df_to_geocode)} 건)")
    print(f"--- 원본 데이터프레임 정보 ({current_csv_file}) ---")
    df_to_geocode.info()
except FileNotFoundError:
    print(f"오류: '{current_csv_file}' 파일을 찾을 수 없습니다. 파일 경로와 이름을 확인해주세요.")
    exit()
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")
    exit()

city_column = 'City'
state_column = 'State'
zip_column = 'Zip Code'

if zip_column in df_to_geocode.columns:
    print(f"\n--- '{zip_column}' 컬럼 고유값 및 개수 (상위 10개) ---")
    print(df_to_geocode[zip_column].value_counts(dropna=False).head(10))
else:
    print(f"경고: '{zip_column}' 컬럼을 찾을 수 없습니다. 'City', 'State' 만으로 Geocoding을 시도합니다.")

geolocator = Nominatim(user_agent="SGO_Crash_Geocoding_Script/1.0")

geocode_with_delay = RateLimiter(geolocator.geocode, min_delay_seconds=1)

df_to_geocode['Approx_Latitude'] = None
df_to_geocode['Approx_Longitude'] = None

print("\n--- Geocoding 시작 (도시/주 기반, 유효한 경우 Zip Code 포함) ---")
for index, row in df_to_geocode.iterrows():
    address_parts = []
    
    if city_column in df_to_geocode.columns and pd.notna(row.get(city_column)):
        address_parts.append(str(row[city_column]))
    
    if state_column in df_to_geocode.columns and pd.notna(row.get(state_column)):
        address_parts.append(str(row[state_column]).strip())
    
    if zip_column in df_to_geocode.columns and pd.notna(row.get(zip_column)):
        zip_value_str = str(row[zip_column])
        if not zip_value_str.startswith('[MAY CONTAIN') and zip_value_str.replace('-', '').isdigit():
            try:
                processed_zip = str(int(float(zip_value_str))).zfill(5)
                address_parts.append(processed_zip)
            except ValueError:
                pass 

    full_address_for_geocoding = ", ".join(address_parts)

    if full_address_for_geocoding:
        try:
            location = geocode_with_delay(full_address_for_geocoding, timeout=10)
            if location:
                df_to_geocode.loc[index, 'Approx_Latitude'] = location.latitude
                df_to_geocode.loc[index, 'Approx_Longitude'] = location.longitude
                if (index + 1) % 20 == 0 or index == 0 :
                     print(f"  진행 ({index + 1}/{len(df_to_geocode)}) -> 좌표: ({location.latitude:.6f}, {location.longitude:.6f}) for {full_address_for_geocoding}")
            else:
                if (index + 1) % 20 == 0 or index == 0 :
                    print(f"  진행 ({index + 1}/{len(df_to_geocode)}) -> 위치를 찾을 수 없음: {full_address_for_geocoding}")
        except Exception as e:
            print(f"  -> Geocoding 오류 ({index + 1}): {e} ({full_address_for_geocoding})")
    else:
        print(f"Geocoding 건너뜀 ({index + 1}/{len(df_to_geocode)}): 주소 정보 부족")
    
print("\n--- Geocoding 완료 ---")
print("\n--- Geocoding 결과 (처음 5개 행의 위도/경도) ---")
cols_to_show_result = [city_column, state_column]
if zip_column in df_to_geocode.columns:
    cols_to_show_result.append(zip_column)
cols_to_show_result.extend(['Approx_Latitude', 'Approx_Longitude'])
valid_cols_to_show_result = [col for col in cols_to_show_result if col in df_to_geocode.columns]
if valid_cols_to_show_result:
    print(df_to_geocode[valid_cols_to_show_result].head())

output_filename = current_csv_file.replace('.csv', '_with_Geocodes.csv')
try:
    df_to_geocode.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\nGeocoding 결과가 '{output_filename}' 파일로 성공적으로 저장되었습니다.")
    print(f"저장된 파일 위치: 스크립트가 실행된 폴더 ({output_filename})")
except Exception as e:
    print(f"\n오류: 파일 저장 중 문제가 발생했습니다 - {e}")