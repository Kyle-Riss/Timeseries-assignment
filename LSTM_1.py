import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # BatchNormalization 추가
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight 
from sklearn.metrics import classification_report, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 

# --- 0. 전처리된 데이터 로드 (이전 코드와 동일) ---
preprocessed_file = 'preprocessed_combined_data_with_weather.csv'
try:
    df_final = pd.read_csv(preprocessed_file)
    print(f"'{preprocessed_file}' 파일 로드 성공. 데이터 형태: {df_final.shape}")
except FileNotFoundError:
    print(f"오류: '{preprocessed_file}' 파일을 찾을 수 없습니다. 날씨 통합 스크립트를 먼저 실행해주세요.")
    exit()

df_final['Incident_Datetime'] = pd.to_datetime(df_final['Incident_Datetime'], errors='coerce')
df_final.dropna(subset=['Incident_Datetime'], inplace=True) 
df_final.sort_values(by='Incident_Datetime', inplace=True) 

# --- 1. 예측 대상 (Target Variable) 정의 및 데이터 전처리 강화 (이전 코드와 동일) ---
target_columns_prefix = 'Highest Injury Severity Alleged_'
target_cols = [col for col in df_final.columns if col.startswith(target_columns_prefix)]

if not target_cols:
    print(f"오류: '{target_columns_prefix}'로 시작하는 예측 대상 컬럼을 찾을 수 없습니다.")
    print("예측 대상을 명확히 정의하고, 전처리 시 해당 컬럼이 원-핫 인코딩되었는지 확인해주세요.")
    exit()

features_df = df_final.drop(columns=['Incident_Datetime'] + target_cols)
target_df = df_final[target_cols]

cols_to_drop_object = []
for col in features_df.columns:
    if features_df[col].dtype == 'object':
        cols_to_drop_object.append(col)
features_df = features_df.drop(columns=cols_to_drop_object, errors='ignore')

for col in features_df.columns:
    if features_df[col].dtype == bool:
        features_df[col] = features_df[col].astype(int)

cols_to_drop_all_nan = features_df.columns[features_df.isna().all()].tolist()
if cols_to_drop_all_nan:
    print(f"제거: 모든 값이 NaN인 컬럼 {cols_to_drop_all_nan}을 제거합니다.")
    features_df = features_df.drop(columns=cols_to_drop_all_nan)
else:
    print("모든 값이 NaN인 컬럼은 없습니다.")

numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols:
    if features_df[col].isna().sum() > 0:
        nan_count = features_df[col].isna().sum()
        median_val = features_df[col].dropna().median() 
        if pd.isna(median_val): median_val = 0.0 
        features_df[col].fillna(median_val, inplace=True)
        print(f"처리: 수치형 컬럼 '{col}'의 NaN 값 ({nan_count}개)을 중앙값({median_val})으로 대체했습니다.")
    
    if np.isinf(features_df[col]).sum() > 0:
        inf_count = np.isinf(features_df[col]).sum()
        median_val_inf = features_df[col].replace([np.inf, -np.inf], np.nan).dropna().median()
        if pd.isna(median_val_inf): median_val_inf = 0.0
        features_df[col].replace([np.inf, -np.inf], median_val_inf, inplace=True)
        print(f"처리: 수치형 컬럼 '{col}'의 Inf 값 ({inf_count}개)을 중앙값({median_val_inf})으로 대체했습니다.")

numerical_cols_for_scaling = []
for col in features_df.columns:
    if pd.api.types.is_numeric_dtype(features_df[col]):
        if features_df[col].nunique() > 2 or features_df[col].max() > 1 or features_df[col].min() < 0:
            numerical_cols_for_scaling.append(col)
            
if numerical_cols_for_scaling:
    scaler = StandardScaler() 
    features_df[numerical_cols_for_scaling] = scaler.fit_transform(features_df[numerical_cols_for_scaling])
    print(f"선택된 수치형 컬럼 {len(numerical_cols_for_scaling)}개 스케일링 완료.")
else:
    print("스케일링할 수치형 컬럼이 없거나 모든 컬럼이 바이너리입니다.")

print(f"\n최종 특성(Features) 개수 (전처리 후): {features_df.shape[1]}")
print(f"최종 특성(Features) DataFrame의 NaN 값 총 개수: {features_df.isna().sum().sum()}")
print(f"최종 특성(Features) DataFrame의 Inf 값 총 개수: {np.isinf(features_df).sum().sum()}")
print(f"최종 타겟(Targets) DataFrame의 NaN 값 총 개수: {target_df.isna().sum().sum()}")


# --- 2. 시퀀스 데이터 생성 함수 (이전 코드와 동일) ---
def create_sequences(features, targets, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features.iloc[i:(i + sequence_length)].values)
        y.append(targets.iloc[i + sequence_length].values)
    return np.array(X), np.array(y)

sequence_length = 10 
X, y = create_sequences(features_df, target_df, sequence_length)

print(f"\n시퀀스 데이터 생성 완료.")
print(f"X (입력 시퀀스) 형태: {X.shape} (샘플 수, 시퀀스 길이, 피처 수)")
print(f"y (예측 대상) 형태: {y.shape} (샘플 수, 타겟 피처 수)")

# --- 3. 데이터셋 분할 (시간 순서 기반, 클래스 분포 확인) (이전 코드와 동일) ---
print("\n--- 3. 데이터셋 분할 (시간 순서 기반, 클래스 분포 확인) ---")

total_samples = X.shape[0]
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_end_idx = int(total_samples * train_ratio)
val_end_idx = train_end_idx + int(total_samples * val_ratio)

X_train, y_train = X[:train_end_idx], y[:train_end_idx]
X_val, y_val = X[train_end_idx:val_end_idx], y[train_end_idx:val_end_idx]
X_test, y_test = X[val_end_idx:], y[val_end_idx:]

print(f"\n데이터 분할 완료 (시간 순서 기반):")
print(f"훈련 세트: {X_train.shape[0]} 샘플")
print(f"검증 세트: {X_val.shape[0]} 샘플")
print(f"테스트 세트: {X_test.shape[0]} 샘플")

y_train_labels_split = np.argmax(y_train, axis=1)
y_val_labels_split = np.argmax(y_val, axis=1)
y_test_labels_split = np.argmax(y_test, axis=1)

print(f"클래스 인덱스 매핑: {list(enumerate(target_cols))}")

print("\n**분할 후 훈련 세트 클래스 분포:**")
print(pd.Series(y_train_labels_split).value_counts().sort_index())
print("\n**분할 후 검증 세트 클래스 분포:**")
print(pd.Series(y_val_labels_split).value_counts().sort_index())
print("\n**분할 후 테스트 세트 클래스 분포:**")
print(pd.Series(y_test_labels_split).value_counts().sort_index())

# --- 클래스 불균형 처리: class_weight 계산 (이전 코드와 동일) ---
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_labels_split), 
    y=y_train_labels_split
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\n계산된 클래스 가중치: {class_weight_dict}")


# --- 4. LSTM 모델 설계 및 학습 (BatchNormalization 적용) ---
print("\n--- 4. LSTM 모델 설계 및 학습 (BatchNormalization 적용 및 평가 지표 확장) ---")

optimizer = Adam(learning_rate=0.0005) # Learning Rate 명시

# 모델 설계에 BatchNormalization 층 추가
units_1 = 100
units_2 = 50
dropout_rate = 0.3 # 다시 기본 드롭아웃으로 시작하여 검증 정확도 변화 관찰

model = Sequential([
    LSTM(units=units_1, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    BatchNormalization(), # 첫 번째 LSTM 층 이후 Batch Normalization 추가
    Dropout(dropout_rate), 
    LSTM(units=units_2, activation='relu', return_sequences=False), 
    BatchNormalization(), # 두 번째 LSTM 층 이후 Batch Normalization 추가
    Dropout(dropout_rate),
    Dense(units=target_df.shape[1], activation='softmax') 
])

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) 

history = model.fit(
    X_train, y_train,
    epochs=200, 
    batch_size=64, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weight_dict, 
    verbose=1 
)

print("\n--- 5. 모델 평가 (상세 지표 포함) ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n테스트 손실 (Loss): {loss:.4f}")
print(f"테스트 정확도 (Accuracy): {accuracy:.4f}")

y_pred_proba = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_proba, axis=1) 
y_true_labels = np.argmax(y_test, axis=1) 

print("\n--- 테스트 세트 분류 보고서 ---")
print(classification_report(y_true_labels, y_pred_labels, 
                            target_names=[col.replace('Highest Injury Severity Alleged_', '') for col in target_cols],
                            labels=range(len(target_cols)))) # 모든 클래스 인덱스를 명시


print("\n--- 테스트 세트 혼동 행렬 ---")
cm = confusion_matrix(y_true_labels, y_pred_labels)
print(cm)

if cm.shape[0] <= 10: 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[target_cols[i].replace('Highest Injury Severity Alleged_', '') for i in np.unique(y_true_labels)],
                yticklabels=[target_cols[i].replace('Highest Injury Severity Alleged_', '') for i in np.unique(y_true_labels)])
    plt.title('Confusion Matrix for Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# --- 학습 커브 시각화 (이전 코드와 동일) ---
print("\n--- 학습 커브 시각화 ---")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='훈련 정확도')
plt.plot(epochs_range, val_acc, label='검증 정확도')
plt.title('훈련 및 검증 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='훈련 손실')
plt.plot(epochs_range, val_loss, label='검증 손실')
plt.title('훈련 및 검증 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

print("학습 커브 시각화 완료.")
print("\nLSTM 모델 학습 및 평가가 완료되었습니다!")
print("Batch Normalization 적용 후 성능 변화를 확인하고, 최종 보고서를 작성할 수 있습니다.")