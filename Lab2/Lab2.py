import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('ggplot')
#Bai 1
df = pd.read_csv(r'C:\LuyenTap\Labs\Lab2\ITA105_Lab_2_Housing.csv')

print("--- 1. Kiểm tra cấu trúc dữ liệu ---")
print(f"Shape: {df.shape}")
print("\nThông tin Missing values:")
print(df.isnull().sum())

print("\n--- 2. Thống kê mô tả ---")
desc = df.describe().T
desc['median'] = df.median()
print(desc[['mean', 'median', 'std', 'min', 'max']])

# Nhận xét sơ bộ: 
# - Mean diện tích (~112) và gia (~1431) khá gần Median -> dữ liệu có vẻ tập trung.
# - Tuy nhiên, Max của diện tích (1000) và gia (50000) lớn hơn rất nhiều so với Mean 
#  Có dấu hiệu của ngoại lệ cực lớn.

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(data=df[['dien_tich', 'gia']], ax=axes[0])
axes[0].set_title("Boxplot Diện tích và Giá (Trước xử lý)")


sns.scatterplot(data=df, x='dien_tich', y='gia', ax=axes[1])
axes[1].set_title("Scatter Plot: Diện tích vs Giá")
plt.show()


print("\n--- 5 & 6. Xác định ngoại lệ ---")

Q1 = df['dien_tich'].quantile(0.25)
Q3 = df['dien_tich'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['dien_tich'] < lower_bound) | (df['dien_tich'] > upper_bound)]

z_scores = np.abs(stats.zscore(df[['dien_tich', 'gia']]))
outliers_z = df[(z_scores > 3).any(axis=1)]

print(f"Số lượng ngoại lệ phát hiện bằng IQR (Diện tích): {len(outliers_iqr)}")
print(f"Số lượng ngoại lệ phát hiện bằng Z-score (|Z|>3): {len(outliers_z)}")

# 8. Phân tích nguyên nhân:
# Quan sát outliers_z: Có các dòng diện tích=1000, gia=50000, so_phong=20.
# Đây khả năng cao là LỖI NHẬP LIỆU (Outlier sai lệch) vì giá trị quá phi thực tế.

df_clean = df.copy()
df_clean['dien_tich'] = df_clean['dien_tich'].clip(lower_bound, upper_bound)

# Với cột 'gia', mình sẽ loại bỏ dòng cực đoan nhất (lỗi nhập liệu)
df_clean = df_clean[df_clean['gia'] < 10000]


sns.boxplot(data=df_clean[['dien_tich', 'gia']])
plt.title("Boxplot Diện tích và Giá (Sau khi xử lý Clip & Filter)")
plt.show()

print("\n--- 10. Nhận xét sau xử lý ---")
print("Sau khi xử lý, thang đo (scale) của biểu đồ đã trở nên hợp lý hơn.")
print("Dữ liệu không còn bị kéo lệch bởi các điểm cực đoan, giúp các mô hình máy học sau này chính xác hơn.")

# Bai 2
#1
df_iot = pd.read_csv(r'C:\LuyenTap\Labs\Lab2\ITA105_Lab_2_Iot.csv')
df_iot['timestamp'] = pd.to_datetime(df_iot['timestamp'])
df_iot.set_index('timestamp', inplace=True)

print("--- 1. Kiểm tra Missing Values ---")
print(df_iot.isnull().sum())

plt.figure(figsize=(15, 6))
for sensor in df_iot['sensor_id'].unique():
    data = df_iot[df_iot['sensor_id'] == sensor]
    plt.plot(data.index, data['temperature'], label=f'Sensor {sensor}')

plt.title('Biến thiên nhiệt độ theo thời gian (Từng Sensor)')
plt.xlabel('Thời gian')
plt.ylabel('Nhiệt độ (°C)')
plt.legend()
plt.show()

print("\n--- 3. Phát hiện ngoại lệ bằng Rolling Mean ---")

s1_data = df_iot[df_iot['sensor_id'] == 'S1']['temperature'].copy()
rolling_mean = s1_data.rolling(window=10).mean()
rolling_std = s1_data.rolling(window=10).std()

upper_bound = rolling_mean + (3 * rolling_std)
lower_bound = rolling_mean - (3 * rolling_std)

outliers_rolling = s1_data[(s1_data > upper_bound) | (s1_data < lower_bound)]
print(f"Số lượng ngoại lệ (Rolling Mean - S1): {len(outliers_rolling)}")
# 4
print("\n--- 4. Phát hiện ngoại lệ bằng Z-Score ---")
df_iot['z_score_temp'] = df_iot.groupby('sensor_id')['temperature'].transform(lambda x: np.abs(stats.zscore(x)))
outliers_z = df_iot[df_iot['z_score_temp'] > 3]
print(f"Số lượng ngoại lệ toàn bộ dataset (|Z| > 3): {len(outliers_z)}")

#5
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.boxplot(data=df_iot, x='sensor_id', y='temperature', ax=axes[0])
axes[0].set_title("Boxplot Nhiệt độ theo từng Sensor")


sns.scatterplot(data=df_iot, x='temperature', y='pressure', hue='sensor_id', alpha=0.5, ax=axes[1])

sns.scatterplot(data=outliers_z, x='temperature', y='pressure', color='red', marker='X', s=100, label='Z-Outlier', ax=axes[1])
axes[1].set_title("Temperature vs Pressure (Highlight Outliers)")
plt.show()

#7
df_clean = df_iot.copy()
df_clean.loc[df_clean['z_score_temp'] > 3, 'temperature'] = np.nan
df_clean['temperature'] = df_clean['temperature'].interpolate(method='time')


plt.figure(figsize=(15, 5))
plt.plot(df_clean[df_clean['sensor_id'] == 'S1'].index, 
         df_clean[df_clean['sensor_id'] == 'S1']['temperature'], color='green')
plt.title("Dữ liệu Sensor S1 sau khi xử lý (Interpolation)")
plt.show()


#Bai 3
df_ecom = pd.read_csv(r'C:\LuyenTap\Labs\Lab2\ITA105_Lab_2_Ecommerce.csv')

print("--- 1. Kiểm tra Missing Values & Thống kê ---")
print(df_ecom.isnull().sum())
print("\nThống kê mô tả:")
print(df_ecom.describe())
#2
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_ecom[['price', 'quantity', 'rating']])
plt.title("Boxplot các biến numeric (Trước xử lý)")
plt.show()

#3
z_scores = np.abs(stats.zscore(df_ecom[['price', 'quantity', 'rating']]))
df_ecom['is_outlier_z'] = (z_scores > 3).any(axis=1)

def get_iqr_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

print(f"\nSố lượng ngoại lệ Z-score: {df_ecom['is_outlier_z'].sum()}")
print(f"Số lượng ngoại lệ IQR (Price): {len(get_iqr_outliers(df_ecom, 'price'))}")

#4
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_ecom[df_ecom['is_outlier_z']==False], x='price', y='quantity', color='blue', label='Normal')
sns.scatterplot(data=df_ecom[df_ecom['is_outlier_z']==True], x='price', y='quantity', color='red', marker='X', s=100, label='Outlier')
plt.title("Scatter Plot: Price vs Quantity (Highlight Outliers)")
plt.legend()
plt.show()

#5,6
# Phân tích: 
# - Price = 0 hoặc Quantity = 0: Lỗi nhập liệu -> Loại bỏ.
# - Rating > 5: Lỗi logic (thang đo chỉ đến 5) -> Loại bỏ.
# - Price rất cao (ví dụ > 500): Có thể là hàng Premium -> Giữ lại nhưng Log-transform.

df_clean = df_ecom[
    (df_ecom['price'] > 0) & 
    (df_ecom['quantity'] > 0) & 
    (df_ecom['rating'] <= 5)
].copy()

df_clean['price_log'] = np.log1p(df_clean['price'])

upper_limit_qty = df_clean['quantity'].quantile(0.99)
df_clean['quantity_clipped'] = df_clean['quantity'].clip(upper=upper_limit_qty)
#7
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(data=df_clean[['price_log', 'quantity_clipped', 'rating']], ax=axes[0])
axes[0].set_title("Boxplot sau xử lý (Log & Clip)")

sns.scatterplot(data=df_clean, x='price_log', y='quantity_clipped', ax=axes[1], alpha=0.5)
axes[1].set_title("Scatter Plot sau xử lý")

plt.show()


#Bai 4
plt.rcParams['figure.figsize'] = (18, 6)
plt.style.use('ggplot')

# 1
df_h = pd.read_csv(r'C:\LuyenTap\Labs\Lab2\ITA105_Lab_2_Housing.csv')
df_iot = pd.read_csv('ITA105_Lab_2_Iot.csv')
df_e = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')


def detect_multivariate_outliers(df, columns):  
    z_scores = np.abs(stats.zscore(df[columns]))
    is_outlier = (z_scores > 3).any(axis=1)
    return is_outlier

df_h['is_outlier'] = detect_multivariate_outliers(df_h, ['dien_tich', 'gia'])
df_iot['is_outlier'] = detect_multivariate_outliers(df_iot, ['temperature', 'pressure'])
df_e['is_outlier'] = detect_multivariate_outliers(df_e, ['price', 'quantity', 'rating'])

#3
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.scatterplot(data=df_h, x='dien_tich', y='gia', hue='is_outlier', 
                palette={True: 'red', False: 'blue'}, ax=axes[0])
axes[0].set_title("Housing: Area vs Price")

sns.scatterplot(data=df_iot, x='temperature', y='pressure', hue='is_outlier', 
                palette={True: 'red', False: 'blue'}, ax=axes[1], alpha=0.5)
axes[1].set_title("IoT: Temp vs Pressure")


sns.scatterplot(data=df_e, x='price', y='quantity', hue='is_outlier', 
                palette={True: 'red', False: 'blue'}, ax=axes[2])
axes[2].set_title("E-commerce: Price vs Qty")

plt.tight_layout()
plt.show()
#4
print("--- 4. Phân tích so sánh ---")
univariate_area = (np.abs(stats.zscore(df_h['dien_tich'])) > 3).sum()
multivariate_h = df_h['is_outlier'].sum()

print(f"Housing - Ngoại lệ đơn biến (chỉ tính diện tích): {univariate_area}")
print(f"Housing - Ngoại lệ đa biến (diện tích + giá): {multivariate_h}")
