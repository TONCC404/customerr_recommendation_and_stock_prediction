import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# Load the client data
client_data_df = pd.read_excel('Client_Details.xlsx', sheet_name='Data')
investment_columns = client_data_df.columns[2:]

# STEP 1: Calculate Cosine Similarity
similarity_matrix = cosine_similarity(client_data_df[investment_columns])
similarity_df = pd.DataFrame(similarity_matrix, index=client_data_df['client_ID'], columns=client_data_df['client_ID'])

# STEP 2: Find Similar Users
most_similar_users = similarity_df.apply(lambda row: row.drop(row.name).idxmax(), axis=1)

# Prepare the recommended portfolio DataFrame
recommended_portfolio_df = most_similar_users.reset_index()
recommended_portfolio_df.columns = ['target_client_ID', 'recommended_from_client_ID']
recommended_portfolio_df = recommended_portfolio_df.merge(client_data_df, left_on='recommended_from_client_ID', right_on='client_ID')
recommended_portfolio_df = recommended_portfolio_df[['target_client_ID', 'recommended_from_client_ID'] + [f'{col}' for col in investment_columns]]
# 计算每隔10个客户的刻度位置和标签
tick_positions = range(0, len(client_data_df['client_ID']), 10)  # 0, 10, 20, 30, ...
tick_labels = client_data_df['client_ID'].iloc[tick_positions]  # 选择每隔10个客户的ID

plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity between Clients')
plt.xlabel('Client ID')
plt.ylabel('Client ID')
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=90)
plt.yticks(ticks=tick_positions, labels=tick_labels)
plt.show()


# 加载数据
data_df = pd.read_excel('ASX200top10.xlsx', sheet_name='Bloomberg raw')

# 选择和重命名相关列
selected_columns = {
    'Unnamed: 0': 'Date',
    'AS51 Index': 'AS51_Return',
    'BHP AT Equity': 'BHP_Return',
    'RIO AT Equity': 'RIO_Return',
    'CBA AT Equity': 'CBA_Return',
    'TLS AT Equity': 'TLS_Return',
    'AMC AT Equity': 'AMC_Return',
    'CSL AT Equity': 'CSL_Return',
    'WOW AT Equity': 'WOW_Return',
    'WES AT Equity': 'WES_Return',
    'FPH AT Equity': 'FPH_Return'
}

returns_columns = list(selected_columns.values())[1:]

# 清理数据
stocks_data = data_df[list(selected_columns.keys())].rename(columns=selected_columns).iloc[1:]
stocks_data = stocks_data[pd.to_datetime(stocks_data['Date'], errors='coerce').notna()]
stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])
stocks_data[returns_columns] = stocks_data[returns_columns].apply(pd.to_numeric, errors='coerce')
stocks_data.set_index('Date', inplace=True)

# 计算平均日收益率和协方差矩阵
mean_daily_returns = stocks_data.mean()
cov_matrix = stocks_data[returns_columns].cov()

# 定义一个函数来计算给定权重下投资组合的风险
def calculate_risk(weights, cov_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    print('22222222222',np.sqrt(portfolio_variance))
    return np.sqrt(portfolio_variance)

# 定义一个函数，返回投资组合风险的平方
def minimize_risk_for_target_return(weights):
    portfolio_return = np.sum(weights * mean_daily_returns)
    constraint = portfolio_return - possible_return
    return 10000 * constraint ** 2 + calculate_risk(weights, cov_matrix)

# 定义约束条件和边界条件
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for asset in range(len(returns_columns)))

# 计算有效前沿
frontier_y = np.linspace(min(mean_daily_returns), max(mean_daily_returns), 100)
frontier_x = []
portfolio_weights = []  # 用于存储投资权重

for possible_return in frontier_y:

    result = minimize(minimize_risk_for_target_return, len(returns_columns) * [1. / len(returns_columns), ], bounds=bounds, constraints=constraints)
    frontier_x.append(calculate_risk(result['x'], cov_matrix))
    portfolio_weights.append(result['x'])


# 存储有效前沿
efficient_frontier_df = pd.DataFrame({
    'Return': frontier_y,
    'Risk': frontier_x
})
# 在efficient_frontier_df中添加权重信息
for i, col in enumerate(returns_columns):
    efficient_frontier_df[f'Weight_{col}'] = [weights[i] for weights in portfolio_weights]

print(efficient_frontier_df.head())

# Load client data
client_df = pd.read_excel('Client_Details.xlsx', sheet_name='Data')

# Extracting the 'risk_profile' column
client_risk_levels = client_df['risk_profile'].astype(int)
client_risk_levels.head()

# Finding the portfolios with minimum and maximum risk on the Efficient Frontier
min_risk_portfolio = efficient_frontier_df.loc[efficient_frontier_df['Risk'].idxmin()]
max_risk_portfolio = efficient_frontier_df.loc[efficient_frontier_df['Risk'].idxmax()]

# Linear interpolation to find the target risk for each risk level
risk_levels = np.arange(1, 10)
interp_risk = interp1d([1, 9], [min_risk_portfolio['Risk'], max_risk_portfolio['Risk']])
target_risks = interp_risk(risk_levels)


# Finding the closest portfolio on the Efficient Frontier for each target risk
selected_portfolios = []
for target_risk in target_risks:
    closest_idx = (efficient_frontier_df['Risk'] - target_risk).abs().idxmin()
    selected_portfolios.append(efficient_frontier_df.loc[closest_idx])

# Creating a DataFrame with the selected portfolios for each risk level
selected_portfolios_df = pd.DataFrame(selected_portfolios)
selected_portfolios_df['Risk Level'] = risk_levels
selected_portfolios_df.set_index('Risk Level', inplace=True)


# 计算有效前沿并存储投资权重
frontier_y = np.linspace(min(mean_daily_returns), max(mean_daily_returns), 100)
frontier_x = []
portfolio_weights = []  # 用于存储投资权重
for possible_return in frontier_y:
    result = minimize(minimize_risk_for_target_return, len(returns_columns) * [1. / len(returns_columns), ], bounds=bounds, constraints=constraints)
    frontier_x.append(calculate_risk(result['x'], cov_matrix))
    portfolio_weights.append(result['x'])

# 在efficient_frontier_df中添加权重信息
for i, col in enumerate(returns_columns):
    efficient_frontier_df[f'Weight_{col}'] = [weights[i] for weights in portfolio_weights]

for i, row in selected_portfolios_df.iterrows():
    print(f"Risk Level {i}:")
    for col in returns_columns:
        print(f"    {col}: {row[f'Weight_{col}'] * 100:.2f}%")

# 为每个客户推荐投资组合
recommendations = []
for index, row in client_df.iterrows():
    risk_level = row['risk_profile']
    recommended_portfolio = selected_portfolios_df.loc[risk_level]
    recommendations.append({
        'Client ID': row['client_ID'],
        'Risk Level': risk_level,
        'Recommended Portfolio': recommended_portfolio
    })

# 打印推荐信息
for rec in recommendations:
    print(f"Client ID {rec['Client ID']} (Risk Level {rec['Risk Level']}):")
    for col in returns_columns:
        print(f"    {col}: {rec['Recommended Portfolio'][f'Weight_{col}'] * 100:.2f}%")


# 客户风险偏好与选定投资组合的风险和收益
plt.figure(figsize=(10,6))
plt.plot(risk_levels, target_risks, label='Target Risks for each Risk Level')
plt.scatter(selected_portfolios_df.index, selected_portfolios_df['Risk'], c='red', label='Selected Portfolio Risk')
plt.title('Client Risk Levels vs Portfolio Risk')
plt.xlabel('Client Risk Level')
plt.ylabel('Portfolio Risk')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# 所有投资组合的风险和收益散点图
plt.figure(figsize=(10,6))
plt.scatter(frontier_x, frontier_y, c='blue', label='All Portfolios')
plt.title('Risk and Return of All Portfolios')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.grid(True)
plt.show()

# 选定投资组合的资产权重
# 假设 result['x'] 包含了选定投资组合的权重
weights = result['x']
plt.figure(figsize=(10,6))
plt.pie(weights, labels=returns_columns, autopct='%1.1f%%')
plt.title('Asset Weights of Selected Portfolio')
plt.show()
