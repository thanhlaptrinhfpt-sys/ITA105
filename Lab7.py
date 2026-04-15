import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from scipy import stats
#1
df = pd.read_csv("C:\Data processing\Labs\Lab7\ITA105_Lab_7.csv")
print(df.head(10))
print(df.shape)

numeric_cols = df.select_dtypes(include=['number'])
skewness_values = numeric_cols.skew()
skew_df = pd.DataFrame(skewness_values, columns=['Skewness'])
skew_df['Abs_Skew'] = skew_df['Skewness'].abs()
top_10_skewed = skew_df.sort_values(by='Abs_Skew', ascending=False).head(10)

top_3_cols = top_10_skewed.index[:3].tolist()
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
for i, col in enumerate(top_3_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='salmon', bins=30)
    skew_val = skewness_values[col]
    axes[i].set_title(f'Phân phối của cột: {col} (Skewness: {skew_val:.2f})', fontsize=14)
    axes[i].set_xlabel('Giá trị')
    axes[i].set_ylabel('Tần suất')
    axes[i].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================================================================
2
df['Negative_Col'] = df['LotArea'] - df['LotArea'].mean()
df['SalePrice_Log'] = np.log1p(df['SalePrice'])
df['LotArea_BoxCox'], lmbda = stats.boxcox(df['LotArea'])
pt = PowerTransformer(method='yeo-johnson')
df['Negative_Col_YJ'] = pt.fit_transform(df[['Negative_Col']])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df['LotArea'], kde=True, ax=axes[0], color='red')
axes[0].set_title("Phân phối LotArea Gốc (Lệch mạnh)")

sns.histplot(df['LotArea_BoxCox'], kde=True, ax=axes[1], color='green')
axes[1].set_title(f"Phân phối LotArea sau Box-Cox (λ={lmbda:.2f})")
plt.show()

#===========================================================================================
#3 
features = ['LotArea', 'NegSkewIncome', 'HouseAge', 'MixedFeature', 'Rooms']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_a = LinearRegression()
model_a.fit(X_train, y_train)

y_train_log = np.log1p(y_train)
model_b = LinearRegression()
model_b.fit(X_train, y_train_log)
pred_b_log = model_b.predict(X_test)
pred_b = np.expm1(pred_b_log)

pt_X = PowerTransformer(method='yeo-johnson')
pt_y = PowerTransformer(method='yeo-johnson')


X_train_pt = pt_X.fit_transform(X_train)
X_test_pt = pt_X.transform(X_test)
y_train_pt = pt_y.fit_transform(y_train.values.reshape(-1, 1))
model_c = LinearRegression()
model_c.fit(X_train_pt, y_train_pt)
# #=====================================================================
#4

df['SalePrice_Transformed'] = np.log1p(df['SalePrice'])
df['LotArea_Transformed'], _ = stats.boxcox(df['LotArea'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
sns.histplot(df['SalePrice'], kde=True, ax=axes[0, 0], color='red')
axes[0, 0].set_title('Phân phối SalePrice (Gốc) - Lệch phải mạnh', fontsize=12)
sns.scatterplot(data=df, x='LotArea', y='SalePrice', ax=axes[0, 1], alpha=0.5)
axes[0, 1].set_title('Mối quan hệ LotArea vs SalePrice (Gốc)', fontsize=12)

sns.histplot(df['SalePrice_Transformed'], kde=True, ax=axes[1, 0], color='green')
axes[1, 0].set_title('Phân phối SalePrice (Log) - Gần chuẩn', fontsize=12)
sns.scatterplot(data=df, x='LotArea_Transformed', y='SalePrice_Transformed', ax=axes[1, 1], alpha=0.5, color='green')
axes[1, 1].set_title('Mối quan hệ LotArea vs SalePrice (Sau biến đổi)', fontsize=12)
plt.tight_layout()
plt.show()

df['LPI'] = df['SalePrice_Transformed'] / df['LotArea_Transformed']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 100))
df['LPI_Score'] = scaler.fit_transform(df[['LPI']])

print(df[['SalePrice', 'LotArea', 'LPI_Score']].head())