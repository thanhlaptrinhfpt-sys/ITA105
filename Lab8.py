import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate

df = pd.read_csv("C:\Data processing\Labs\Lab8\ITA105_Lab_8.csv")
print(df.head(10))
print(df.shape)
print(df.info())


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.to_datetime(X.iloc[:, 0])
        df_date = pd.DataFrame()
        df_date['month'] = X.dt.month
        df_date['quarter'] = X.dt.quarter
        df_date['year'] = X.dt.year
        return df_date.ffill().bfill()


num_features = ['LotArea', 'Rooms', 'NoiseFeature']
cat_features = ['Neighborhood', 'Condition', 'HasGarage']
text_feature = 'Description'
date_feature = ['SaleDate']


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson'))
])


cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=50))
])


date_pipeline = Pipeline([
    ('extractor', DateFeatureExtractor())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
        ('text', text_pipeline, text_feature),
        ('date', date_pipeline, date_feature)
    ]
)

full_pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

data = {
    'LotArea': [2346, 15050, 6583, 4564, 848, 847, 299, 10056, 4595, 6156],
    'SalePrice': [271382, 551212, 325180, 215767, 219120, 249373, 237866, 187324, 171459, 157740],
    'Rooms': [3, 5, 4, 3, 2, 3, 2, 4, 3, 3],
    'HasGarage': [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    'NoiseFeature': [0.5, 1.2, 0.1, 0.8, 0.3, 0.2, 0.9, 0.4, 0.1, 0.6],
    'Neighborhood': ['CollgCr', 'Veenker', 'CollgCr', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide'],
    'Condition': ['Norm', 'Feedr', 'Norm', 'Norm', 'Norm', 'Norm', 'Norm', 'Norm', 'Artery', 'Artery'],
    'Description': ["Beautiful house with garden", "Luxury villa", "Near park", "Old style", "Modern apartment", "Quiet area", "Big garage", "New roof", "Close to school", "Family home"],
    'SaleDate': ['2009-06-13', '2003-07-24', '2007-04-07', '2003-12-25', '2003-02-12', '2015-11-25', '2008-09-27', '2012-08-25', '2016-09-29', '2013-09-03']
}
df_demo = pd.DataFrame(data)

X_processed = full_pipeline.fit_transform(df_demo)

cat_names = full_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(cat_features)
text_names = full_pipeline.named_steps['preprocessor'].transformers_[2][1].named_steps['tfidf'].get_feature_names_out()
date_names = ['month', 'quarter', 'year']

final_columns = num_features + list(cat_names) + list(text_names) + date_names

print(f"Hình dạng dữ liệu sau Pipeline: {X_processed.shape}")
print("-" * 30)
print("SCHEMA CUỐI CÙNG (Một vài cột tiêu biểu):")
print(final_columns[:15])

#=========================================================
#2
base_data = df_demo.copy()

test_cases = {
    "Full Data": base_data,
    
    "Missing Data": base_data.copy().mask(np.random.random(base_data.shape) < 0.3),
    
    "Skewed Data": base_data.assign(LotArea=[10, 20, 15, 12, 11, 14, 10, 100000, 200000, 150000]),
    
    "Unseen Category": base_data.assign(Neighborhood=['Mars', 'Moon', 'Sun', 'Venus', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Earth']),
    
    "Wrong Format": base_data.assign(Rooms=['Ba', 'Bốn', 'Năm', '2', '3', '4', '5', '6', '7', '8'])
}

results = {}

for name, test_df in test_cases.items():
    print(f"--- Đang kiểm tra: {name} ---")
    try:
        
        if 'SaleDate' in test_df.columns:
            test_df['SaleDate'] = pd.to_datetime(test_df['SaleDate'], errors='coerce').fillna(pd.Timestamp('2026-01-01'))
            
        processed = full_pipeline.transform(test_df)
        results[name] = {"Status": "Thành công", "Shape": processed.shape}
    except Exception as e:
        results[name] = {"Status": "Lỗi", "Error": str(e)}


print(pd.DataFrame(results).T)

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
sns.histplot(df_demo['LotArea'], kde=True, color='red')
plt.title("Phân phối LotArea Trước Pipeline")


processed_all = full_pipeline.fit_transform(df_demo)

plt.subplot(1, 2, 2)
sns.histplot(processed_all[:, 0], kde=True, color='green')
plt.title("Phân phối LotArea Sau Pipeline (Đã chuẩn hóa)")

plt.tight_layout()
plt.show()

#=====================================================================
#3
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

models = {
    "Linear Regression": lr_pipeline,
    "Random Forest": rf_pipeline,
    "XGBoost": xgb_pipeline
}

cv_results = {}

for name, pipe in models.items():
   
    scores = cross_validate(pipe, df_demo, df_demo['SalePrice'], cv=5,
                            scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
    
    cv_results[name] = {
        "RMSE": -scores['test_neg_root_mean_squared_error'].mean(),
        "MAE": -scores['test_neg_mean_absolute_error'].mean(),
        "R2": scores['test_r2'].mean(),
        "Std RMSE": scores['test_neg_root_mean_squared_error'].std() # Mức ổn định
    }


df_compare = pd.DataFrame(cv_results).T
print(df_compare)


rf_pipeline.fit(df_demo, df_demo['SalePrice'])
importances = rf_pipeline.named_steps['regressor'].feature_importances_

feat_imp_df = pd.DataFrame({'Feature': final_columns, 'Importance': importances})
print(feat_imp_df.sort_values(by='Importance', ascending=False).head(10))

#============================================
#4

final_model_pipeline = xgb_pipeline 
final_model_pipeline.fit(df_demo, df_demo['SalePrice'])

joblib.dump(final_model_pipeline, 'house_price_pipeline.pkl')
print("Đã lưu pipeline thành công!")

def predict_price(new_data_path):
    loaded_pipe = joblib.load('house_price_pipeline.pkl')
    new_data = pd.read_csv(new_data_path)
    num_cols = ['LotArea', 'Rooms', 'NoiseFeature']

    for col in num_cols:
        if col in new_data.columns:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    predictions = loaded_pipe.predict(new_data)
    new_data['Predicted_Price'] = predictions

    return new_data[['LotArea', 'Neighborhood', 'Predicted_Price']]

fake_new_customer = pd.DataFrame([{
    'LotArea': 5000,
    'Rooms': 4,
    'NoiseFeature': 0.2,
    'Neighborhood': 'Hanoi_City', 
    'Condition': 'Excellent',    
    'HasGarage': 1,
    'Description': 'Modern house with smart home system',
    'SaleDate': '2026-04-15'
}])


try:
    price = final_model_pipeline.predict(fake_new_customer)
    print(f"Giá nhà dự báo cho khách hàng: {price[0]:,.2f} USD")
except Exception as e:
    print(f"Hệ thống gặp lỗi: {e}")