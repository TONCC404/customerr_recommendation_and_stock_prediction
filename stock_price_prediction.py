import pandas as pd
from datetime import datetime
from economic_indicator_analyze import economic_indicator_analyze
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np



def ASX200top10():
    # 读取Excel文件
    filename = 'ASX200top10.xlsx'  # 请用您的文件路径替换这个路径
    xl = pd.ExcelFile(filename)

    # 从Excel文件中读取特定的表格
    df_raw = xl.parse('Bloomberg raw')

    # 定义感兴趣的公司和指标
    companies = [
        "AS51 Index", "BHP AT Equity", "CSL AT Equity",
        "RIO AT Equity", "CBA AT Equity", "WOW AT Equity",
        "WES AT Equity", "TLS AT Equity", "AMC AT Equity"
    ]

    indicators = [
        "DAY_TO_DAY_TOT_RETURN_GROSS_DVDS", "PX_LAST",
        "EQY_WEIGHTED_AVG_PX", "PX_VOLUME", "CUR_MKT_CAP"
    ]

    # 定义感兴趣的日期范围
    start_date = datetime(2020, 2, 1)
    end_date = datetime(2020, 3, 31)

    # 根据日期范围过滤数据
    df_raw['Unnamed: 0'] = pd.to_datetime(df_raw['Unnamed: 0'], errors='coerce')
    df_filtered = df_raw[(df_raw['Unnamed: 0'] >= start_date) & (df_raw['Unnamed: 0'] <= end_date)]

    # 准备最终的DataFrame，只选择需要的公司和指标
    df_final = pd.DataFrame()
    df_final['Dates'] = df_filtered['Unnamed: 0']

    for company in companies:
        for indicator in indicators:
            column_name = f"{company} {indicator}"
            company_col_index = df_raw.columns.get_loc(company)

            # 找到特定指标的列索引
            indicator_index = 0
            for idx, cell in enumerate(df_raw.iloc[0, company_col_index+1:]):
                if cell == indicator:
                    indicator_index = idx + 1
                    break

            # 将数据添加到最终的DataFrame中
            df_final[column_name] = df_filtered.iloc[:, company_col_index + indicator_index]
    return df_final
    # 显示最终DataFrame的前几行
    # print(df_final,type(df_final))
df_final = ASX200top10()
df_economic = economic_indicator_analyze()


df_combined = pd.concat([df_final.reset_index(drop=True), df_economic.reset_index(drop=True)], axis=1)
for col in df_combined.columns:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

#
y = df_combined['BHP AT Equity CUR_MKT_CAP'].values
X = df_combined[['AS51 Index DAY_TO_DAY_TOT_RETURN_GROSS_DVDS',
       'AS51 Index PX_LAST','BHP AT Equity DAY_TO_DAY_TOT_RETURN_GROSS_DVDS',
       'BHP AT Equity PX_LAST', 'BHP AT Equity EQY_WEIGHTED_AVG_PX',
       'BHP AT Equity PX_VOLUME','CPI (%q/q)',
                'CPI, TD-MI Inflation Gauge Idx (%m/m)', 'Money Supply, M1 (%y/y)',
                'Money Supply, M3 (%y/y)', 'PPI (%q/q)', 'Real GDP Growth (%q/q, sa)',
                'Unemployment Rate (sa)']].values

# 添加一个常数项（截距）
X = sm.add_constant(X)

# 创建模型并拟合数据
model = sm.OLS(y, X).fit()

# 显示回归结果的摘要
print(model.summary())

predictions = model.predict(X)
print(len(predictions))

import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建 XGBoost 回归模型实例
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror')

# 使用数据训练模型
# X_train, y_train 是训练数据的特征和目标变量
xgb_model.fit(X_train, y_train)

# 对测试数据进行预测
# dtest 是一个包含测试集特征的 xgboost DMatrix 对象
# 或者，如果您有一个 DataFrame 或 NumPy 数组，您可以直接使用它进行预测
predictions = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')
# print(predictions)

