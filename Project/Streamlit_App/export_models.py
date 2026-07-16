import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "All_Data_set_Tourism")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

df1 = pd.read_csv(os.path.join(DATA_DIR, "India-Tourism-Statistics-1981-2020-fta_nri_ita.csv"))
df2 = pd.read_csv(os.path.join(DATA_DIR, "India-Tourism-Statistics-2001-2019-agegroup.csv"))
df3 = pd.read_csv(os.path.join(DATA_DIR, "India-Tourism-Statistics-2001-2019-quaterly.csv"))
df4 = pd.read_csv(os.path.join(DATA_DIR, "India-Tourism-Statistics-2001-2019-worldvsindia.csv"))
df5 = pd.read_csv(os.path.join(DATA_DIR, "India-Tourism-Statistics-2019_region-and-reason.csv"))
df6 = pd.read_csv(os.path.join(DATA_DIR, "India-Tourism-Statistics-2021-monuments.csv"))
df7 = pd.read_csv(os.path.join(DATA_DIR, "India-Tourism-Statistics-region-2017-2019.csv"))
df8 = pd.read_csv(os.path.join(DATA_DIR, "India-Tourism-Statistics-statewise_2019-2020_domestic_foreign.csv"))

# ---------- df4 cleaning ----------
df4["Rank of India"] = df4["Rank of India"].str.extract(r'(\d+)').astype(float)
df4["Rank of India"] = df4["Rank of India"].interpolate(method="linear", limit_direction="forward")
df4["Rank of India"] = df4["Rank of India"].astype(int)

# ---------- df6 KNN imputation ----------
x6 = df6.iloc[:, 2:]
imputer6 = KNNImputer(n_neighbors=2)
x6_imputed = imputer6.fit_transform(x6)
df6.iloc[:, 2:] = x6_imputed

# ---------- df7 KNN imputation ----------
x7 = df7.iloc[:, 2:]
imputer7 = KNNImputer(n_neighbors=3)
x7_imputed = imputer7.fit_transform(x7)
df7.iloc[:, 2:] = x7_imputed.astype(int)

# ---------- df8 type conversions ----------
df8['Growth rate - DTV  2020/19'] = df8['Growth rate - DTV  2020/19'].astype(str).str.replace('%', '', regex=False).astype(float)
df8['Growth rate - FTV 2020/19'] = df8['Growth rate - FTV 2020/19'].astype(str).str.replace('%', '', regex=False).astype(float)
df8 = df8[df8['S. No.'].astype(str).str.match(r'^\d+$')]
df8['S. No.'] = df8['S. No.'].astype(int)

# ---------- IQR capping ----------
def cap_iqr(df, cols):
    for i in cols:
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[i] = np.where(df[i] > upper, upper, df[i])
        df[i] = np.where(df[i] < lower, lower, df[i])

cap_iqr(df1, ['FTAs in India (in million)', 'NRIs arrivals in India (in million)', 'ITAs in India  (in million)',
              '%- change over previous year', '%- change over the previous year', '%- change over the previous year.1'])

cap_iqr(df2, ['% distribution by Age-Group (in Year) - 0-14', '% distribution by Age-Group (in Year) - 15-24',
              '% distribution by Age-Group (in Year) - 25-34', '% distribution by Age-Group (in Year) - 35-44',
              '% distribution by Age-Group (in Year) - 45-54', '% distribution by Age-Group (in Year) - 65 and above',
              '% distribution by Age-Group (in Year) - Not Reported'])

cap_iqr(df3, ['Year', 'Arrivals', '% Distribution by Quarter - 1st Quarter (Jan-Mar)',
              '% Distribution by Quarter - 2nd Quarter(Apr-June)', '% Distribution by Quarter - 3rd Quarter (July-Sep)',
              '% Distribution by Quarter - 4th Quarter (Oct-Dec)'])

cap_iqr(df4, ['Year', 'World - Number (in million)', 'World - % Change', 'India - Number (in million)',
              'India - % Change', 'Percentage Share of India', 'Rank of India'])

cap_iqr(df5, ['Arrivals (in numbers)', 'Business and Professional(%)', 'Leisure Holiday and Recreation(%)',
              'Medical(%)', 'Indian Diaspora(%)', 'Others(%)'])

cap_iqr(df7, ['Number of Arrivals - 2017', 'Number of Arrivals - 2018', 'Number of Arrivals - 2019',
              'Percentage Share - 2017', 'Percentage Share - 2018', 'Percentage Share - 2019',
              'Percentage Change - 2018/17', 'Percentage Change - 2019/18'])

cap_iqr(df8, ['S. No.', 'Domestic -2019', 'Foreign - 2019', 'Domestic -2020', 'Foreign - 2020',
              'Growth rate - DTV  2020/19', 'Growth rate - FTV 2020/19'])

# ---------- Drop columns ----------
df1 = df1.drop(columns=['%- change over previous year', 'NRIs arrivals in India (in million)',
                        '%- change over the previous year', 'ITAs in India  (in million)',
                        '%- change over the previous year.1'])
df2 = df2.drop(columns=['% distribution by Age-Group (in Year) - Not Reported',
                        '% distribution by Age-Group (in Year) - 65 and above'])
df3 = df3.drop(columns=['% Distribution by Quarter - 3rd Quarter (July-Sep)'])
df4 = df4.drop(columns=['World - % Change', 'India - % Change', 'Percentage Share of India'])
df5 = df5.drop(columns=['Others(%)'])
df6 = df6.drop(columns=['% Growth 2021-21/2019-20-Domestic', '% Growth 2021-21/2019-20-Foreign'])
df7 = df7.drop(columns=['Percentage Share - 2017', 'Percentage Share - 2018', 'Percentage Share - 2019',
                         'Percentage Change - 2018/17', 'Percentage Change - 2019/18'])

# ---------- Feature engineering ----------
freq_map5_region = df5['Region'].value_counts()
df5['Region_cat'] = df5['Region'].map(freq_map5_region)
df5 = df5.drop(columns=['Region'])

freq_map5_country = df5['Country of Nationality'].value_counts()
df5['Country of Nationality_cat'] = df5['Country of Nationality'].map(freq_map5_country)
df5 = df5.drop(columns=['Country of Nationality'])

freq_map6 = df6['Circle'].value_counts()
df6['Circle_new'] = df6['Circle'].map(freq_map6)
df6 = df6.drop(columns=['Circle'])

le6 = LabelEncoder()
df6['Name of the Monument_new'] = le6.fit_transform(df6['Name of the Monument'])
df6 = df6.drop(columns=['Name of the Monument'])

df7 = pd.get_dummies(df7, columns=['Region'], drop_first=True, dtype=int)

freq_map7 = df7['Country of Nationality'].value_counts()
df7['Country of Nationality_new'] = df7['Country of Nationality'].map(freq_map7)
df7 = df7.drop(columns=['Country of Nationality'])

le8 = LabelEncoder()
df8['States/UTs_new'] = le8.fit_transform(df8['States/UTs'])
df8 = df8.drop(columns=['States/UTs'])

# ---------- df8 extra feature ----------
df8['Total_2019'] = (df8['Domestic -2019'] + df8['Foreign - 2019']).astype(int)
df8['Total_2020'] = (df8['Domestic -2020'] + df8['Foreign - 2020']).astype(int)
df8 = df8.drop(columns=['Domestic -2019', 'Foreign - 2019', 'Domestic -2020', 'Foreign - 2020'])

# ========== MODEL TRAINING ==========
seed = 42

# ---- df1 ----
X1 = df1.drop(columns=['FTAs in India (in million)'])
y1 = df1['FTAs in India (in million)']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=seed)
scaler1 = StandardScaler()
X1_train_s = scaler1.fit_transform(X1_train)
X1_test_s = scaler1.transform(X1_test)

lr1 = LinearRegression().fit(X1_train_s, y1_train)
dt1 = DecisionTreeRegressor(random_state=seed).fit(X1_train_s, y1_train)
rf1 = RandomForestRegressor(random_state=seed).fit(X1_train_s, y1_train)
svr1 = SVR().fit(X1_train_s, y1_train)

grid1 = GridSearchCV(RandomForestRegressor(random_state=seed),
                     {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None],
                      "min_samples_leaf": [1, 2, 4]}, cv=3, scoring="r2")
grid1.fit(X1_train_s, y1_train)
best1 = grid1.best_estimator_

joblib.dump(lr1, os.path.join(MODELS_DIR, "df1_lr.pkl"))
joblib.dump(dt1, os.path.join(MODELS_DIR, "df1_dt.pkl"))
joblib.dump(rf1, os.path.join(MODELS_DIR, "df1_rf.pkl"))
joblib.dump(svr1, os.path.join(MODELS_DIR, "df1_svr.pkl"))
joblib.dump(best1, os.path.join(MODELS_DIR, "df1_best.pkl"))
joblib.dump(scaler1, os.path.join(MODELS_DIR, "scaler1.pkl"))
print(f"df1 best params: {grid1.best_params_}")

# ---- df2 ----
X2 = df2.drop(columns=['% distribution by Age-Group (in Year) - 15-24',
                       '% distribution by Age-Group (in Year) - 25-34'])
y2 = df2[['% distribution by Age-Group (in Year) - 15-24',
          '% distribution by Age-Group (in Year) - 25-34']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=seed)
scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s = scaler2.transform(X2_test)

lr2 = LinearRegression().fit(X2_train_s, y2_train)
dt2 = DecisionTreeRegressor(random_state=seed).fit(X2_train_s, y2_train)
rf2 = RandomForestRegressor(random_state=seed).fit(X2_train_s, y2_train)
svr2 = MultiOutputRegressor(SVR()).fit(X2_train_s, y2_train)

grid2 = GridSearchCV(RandomForestRegressor(random_state=seed),
                     {"n_estimators": [50, 60, 70], "max_depth": [3, 5, None],
                      "min_samples_leaf": [1, 2, 4]}, cv=3, scoring="r2")
grid2.fit(X2_train_s, y2_train)
best2 = grid2.best_estimator_

joblib.dump(lr2, os.path.join(MODELS_DIR, "df2_lr.pkl"))
joblib.dump(dt2, os.path.join(MODELS_DIR, "df2_dt.pkl"))
joblib.dump(rf2, os.path.join(MODELS_DIR, "df2_rf.pkl"))
joblib.dump(svr2, os.path.join(MODELS_DIR, "df2_svr.pkl"))
joblib.dump(best2, os.path.join(MODELS_DIR, "df2_best.pkl"))
joblib.dump(scaler2, os.path.join(MODELS_DIR, "scaler2.pkl"))
print(f"df2 best params: {grid2.best_params_}")

# ---- df3 ----
X3 = df3.drop(columns=['% Distribution by Quarter - 2nd Quarter(Apr-June)',
                       '% Distribution by Quarter - 4th Quarter (Oct-Dec)'])
y3 = df3[['% Distribution by Quarter - 2nd Quarter(Apr-June)',
          '% Distribution by Quarter - 4th Quarter (Oct-Dec)']]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=seed)
scaler3 = StandardScaler()
X3_train_s = scaler3.fit_transform(X3_train)
X3_test_s = scaler3.transform(X3_test)

lr3 = LinearRegression().fit(X3_train_s, y3_train)
dt3 = DecisionTreeRegressor(random_state=seed).fit(X3_train_s, y3_train)
rf3 = RandomForestRegressor(random_state=seed).fit(X3_train_s, y3_train)
svr3 = MultiOutputRegressor(SVR()).fit(X3_train_s, y3_train)

grid3 = GridSearchCV(RandomForestRegressor(random_state=seed),
                     {"n_estimators": [50, 100, 150, 200], "max_depth": [3, 5, None],
                      "min_samples_leaf": [1, 2, 4]}, cv=3, scoring="r2")
grid3.fit(X3_train_s, y3_train)
best3 = grid3.best_estimator_

joblib.dump(lr3, os.path.join(MODELS_DIR, "df3_lr.pkl"))
joblib.dump(dt3, os.path.join(MODELS_DIR, "df3_dt.pkl"))
joblib.dump(rf3, os.path.join(MODELS_DIR, "df3_rf.pkl"))
joblib.dump(svr3, os.path.join(MODELS_DIR, "df3_svr.pkl"))
joblib.dump(best3, os.path.join(MODELS_DIR, "df3_best.pkl"))
joblib.dump(scaler3, os.path.join(MODELS_DIR, "scaler3.pkl"))
print(f"df3 best params: {grid3.best_params_}")

# ---- df4 ----
X4 = df4.drop(columns=['World - Number (in million)', 'India - Number (in million)'])
y4 = df4[['World - Number (in million)', 'India - Number (in million)']]
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=seed)
scaler4 = StandardScaler()
X4_train_s = scaler4.fit_transform(X4_train)
X4_test_s = scaler4.transform(X4_test)

grid4 = GridSearchCV(MultiOutputRegressor(SVR()),
                     {"estimator__kernel": ["linear", "rbf"], "estimator__C": [0.1, 1, 10],
                      "estimator__epsilon": [0.01, 0.1, 0.5]}, cv=3, scoring="r2")
grid4.fit(X4_train_s, y4_train)
best4 = grid4.best_estimator_

joblib.dump(best4, os.path.join(MODELS_DIR, "df4_best.pkl"))
joblib.dump(scaler4, os.path.join(MODELS_DIR, "scaler4.pkl"))
print(f"df4 best params: {grid4.best_params_}")

# ---- df5 ----
X5 = df5.drop(columns=['Business and Professional(%)', 'Leisure Holiday and Recreation(%)',
                       'Medical(%)', 'Indian Diaspora(%)'])
y5 = df5[['Business and Professional(%)', 'Leisure Holiday and Recreation(%)',
          'Medical(%)', 'Indian Diaspora(%)']]
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2, random_state=seed)

lr5 = LinearRegression().fit(X5_train, y5_train)
dt5 = DecisionTreeRegressor(random_state=seed).fit(X5_train, y5_train)
rf5 = RandomForestRegressor(random_state=seed).fit(X5_train, y5_train)
svr5 = MultiOutputRegressor(SVR()).fit(X5_train, y5_train)

grid5 = GridSearchCV(RandomForestRegressor(random_state=seed),
                     {"n_estimators": [50, 100, 150, 200], "max_depth": [3, 5, None],
                      "min_samples_leaf": [1, 2, 4]}, cv=3, scoring="r2")
grid5.fit(X5_train, y5_train)
best5 = grid5.best_estimator_

joblib.dump(lr5, os.path.join(MODELS_DIR, "df5_lr.pkl"))
joblib.dump(dt5, os.path.join(MODELS_DIR, "df5_dt.pkl"))
joblib.dump(rf5, os.path.join(MODELS_DIR, "df5_rf.pkl"))
joblib.dump(svr5, os.path.join(MODELS_DIR, "df5_svr.pkl"))
joblib.dump(best5, os.path.join(MODELS_DIR, "df5_best.pkl"))
print(f"df5 best params: {grid5.best_params_}")

# ---- df6 ----
X6 = df6.drop(columns=['Domestic-2020-21', 'Foreign-2020-21'])
y6 = df6[['Domestic-2020-21', 'Foreign-2020-21']]
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.2, random_state=seed)

lr6 = LinearRegression().fit(X6_train, y6_train)
dt6 = DecisionTreeRegressor(random_state=seed).fit(X6_train, y6_train)
rf6 = RandomForestRegressor(random_state=seed).fit(X6_train, y6_train)
svr6 = MultiOutputRegressor(SVR()).fit(X6_train, y6_train)

grid6 = GridSearchCV(RandomForestRegressor(random_state=seed),
                     {"n_estimators": [150, 200, 250, 300], "max_depth": [15, 20, 25, None],
                      "min_samples_leaf": [1, 2, 5], "min_samples_split": [2, 4, 6],
                      "max_features": [0.6, 0.8, None]}, cv=5, scoring="r2", n_jobs=-1)
grid6.fit(X6_train, y6_train)
best6 = grid6.best_estimator_

joblib.dump(lr6, os.path.join(MODELS_DIR, "df6_lr.pkl"))
joblib.dump(dt6, os.path.join(MODELS_DIR, "df6_dt.pkl"))
joblib.dump(rf6, os.path.join(MODELS_DIR, "df6_rf.pkl"))
joblib.dump(svr6, os.path.join(MODELS_DIR, "df6_svr.pkl"))
joblib.dump(best6, os.path.join(MODELS_DIR, "df6_best.pkl"))
print(f"df6 best params: {grid6.best_params_}")

# ---- df7 ----
X7 = df7[['Number of Arrivals - 2017', 'Number of Arrivals - 2018',
          'Region_Australasia', 'Region_Central and South America', 'Region_East Asia',
          'Region_Eastern Europe', 'Region_North America', 'Region_Not Classified elsewhere',
          'Region_South Asia', 'Region_South East Asia', 'Region_West Asia',
          'Region_Western Europe', 'Country of Nationality_new']]
y7 = df7['Number of Arrivals - 2019']
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.2, random_state=seed)

lr7 = LinearRegression().fit(X7_train, y7_train)
dt7 = DecisionTreeRegressor(random_state=seed).fit(X7_train, y7_train)
rf7 = RandomForestRegressor(random_state=seed).fit(X7_train, y7_train)
svr7 = SVR().fit(X7_train, y7_train)

final7 = LinearRegression().fit(X7_train, y7_train)

joblib.dump(lr7, os.path.join(MODELS_DIR, "df7_lr.pkl"))
joblib.dump(dt7, os.path.join(MODELS_DIR, "df7_dt.pkl"))
joblib.dump(rf7, os.path.join(MODELS_DIR, "df7_rf.pkl"))
joblib.dump(svr7, os.path.join(MODELS_DIR, "df7_svr.pkl"))
joblib.dump(final7, os.path.join(MODELS_DIR, "df7_best.pkl"))

# ---- df8 ----
X8 = df8[['States/UTs_new', 'Total_2019']]
y8 = df8['Total_2020']
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size=0.2, random_state=seed)
scaler8 = StandardScaler()
X8_train_s = scaler8.fit_transform(X8_train)
X8_test_s = scaler8.transform(X8_test)

lr8 = LinearRegression().fit(X8_train_s, y8_train)
dt8 = DecisionTreeRegressor(random_state=seed).fit(X8_train_s, y8_train)
rf8 = RandomForestRegressor(random_state=seed).fit(X8_train_s, y8_train)
svr8 = SVR().fit(X8_train_s, y8_train)

grid8 = GridSearchCV(RandomForestRegressor(random_state=seed),
                     {"n_estimators": [100], "max_depth": [10], "min_samples_leaf": [2],
                      "min_samples_split": [2], "max_features": [None]}, cv=5, scoring="r2")
grid8.fit(X8_train_s, y8_train)
best8 = grid8.best_estimator_

joblib.dump(lr8, os.path.join(MODELS_DIR, "df8_lr.pkl"))
joblib.dump(dt8, os.path.join(MODELS_DIR, "df8_dt.pkl"))
joblib.dump(rf8, os.path.join(MODELS_DIR, "df8_rf.pkl"))
joblib.dump(svr8, os.path.join(MODELS_DIR, "df8_svr.pkl"))
joblib.dump(best8, os.path.join(MODELS_DIR, "df8_best.pkl"))
joblib.dump(scaler8, os.path.join(MODELS_DIR, "scaler8.pkl"))
print(f"df8 best params: {grid8.best_params_}")

# ---------- Save label encoders and frequency maps ----------
joblib.dump(le6, os.path.join(MODELS_DIR, "le6.pkl"))
joblib.dump(le8, os.path.join(MODELS_DIR, "le8.pkl"))
joblib.dump(freq_map5_region, os.path.join(MODELS_DIR, "freq_map5_region.pkl"))
joblib.dump(freq_map5_country, os.path.join(MODELS_DIR, "freq_map5_country.pkl"))
joblib.dump(freq_map6, os.path.join(MODELS_DIR, "freq_map6.pkl"))
joblib.dump(freq_map7, os.path.join(MODELS_DIR, "freq_map7.pkl"))

# ---------- Save column info for each dataset ----------
joblib.dump(list(X1.columns), os.path.join(MODELS_DIR, "cols_X1.pkl"))
joblib.dump(list(X2.columns), os.path.join(MODELS_DIR, "cols_X2.pkl"))
joblib.dump(list(X3.columns), os.path.join(MODELS_DIR, "cols_X3.pkl"))
joblib.dump(list(X4.columns), os.path.join(MODELS_DIR, "cols_X4.pkl"))
joblib.dump(list(X5.columns), os.path.join(MODELS_DIR, "cols_X5.pkl"))
joblib.dump(list(X6.columns), os.path.join(MODELS_DIR, "cols_X6.pkl"))
joblib.dump(list(X7.columns), os.path.join(MODELS_DIR, "cols_X7.pkl"))
joblib.dump(list(X8.columns), os.path.join(MODELS_DIR, "cols_X8.pkl"))

print("\nAll models and artifacts exported successfully to:", MODELS_DIR)
