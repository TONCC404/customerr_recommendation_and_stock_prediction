import pandas as pd  # 导入Pandas库，用于数据处理和分析
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# 加载Excel文件中的数据
data_df = pd.read_excel('ASX200top10.xlsx', sheet_name='Bloomberg raw')

# 选择和重命名相关列。手动选择了包含日收益率数据的列，并将这些列重命名以便于理解。
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

# 选择和重命名列，并排除第一行，因为它包含了额外的表头信息
stocks_data = data_df[list(selected_columns.keys())].rename(columns=selected_columns).iloc[1:]

# 清理数据。将日期列中不是有效日期的行排除，并将其它非数值数据转换为NaN
stocks_data = stocks_data[pd.to_datetime(stocks_data['Date'], errors='coerce').notna()]
stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])
returns_columns = list(selected_columns.values())[1:]
stocks_data[returns_columns] = stocks_data[returns_columns].apply(pd.to_numeric, errors='coerce')

# 将日期列设为索引，以便于后续的时间序列分析
stocks_data.set_index('Date', inplace=True)

# 计算每个股票的平均日收益率和标准差
mean_daily_returns = stocks_data.mean()
std_daily_returns = stocks_data.std()

# 定义无风险收益率为2%
risk_free_rate = 0.02


# Number of portfolios to simulate
num_portfolios = 10000
results = np.zeros((4, num_portfolios))
np.random.seed(42)  # for reproducibility

for i in range(num_portfolios):
    # Generate random weights and normalize to 1
    weights = np.random.random(len(returns_columns))
    weights /= np.sum(weights)

    # Calculate portfolio return and standard deviation
    portfolio_return = np.sum(weights * mean_daily_returns)
    # 计算协方差矩阵
    cov_matrix = stocks_data[returns_columns].cov()

    # 计算投资组合的方差和标准差
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_stddev = np.sqrt(portfolio_variance)

    # Calculate Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

    # Store results
    results[0,i] = portfolio_return
    results[1,i] = portfolio_stddev
    results[2,i] = sharpe_ratio
    results[3,i] = weights[0]  # store the weight of the first stock for illustration

# Extract results and plot
returns, std_devs, sharpe_ratios, weights_first_stock = results

plt.scatter(std_devs, returns, c=sharpe_ratios, cmap='YlGnBu', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.title('Sharpe Ratio of Portfolios with Different Weights')
plt.xlabel('Portfolio Standard Deviation')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.show()

# 找到给定收益下的最小风险投资组合
frontier_y = np.linspace(min(returns), max(returns), 100)  # 100 points between min and max returns
frontier_x = []

# 定义一个函数来计算给定权重下投资组合的风险
def calculate_risk(weights, cov_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(portfolio_variance)

for possible_return in frontier_y:
    # 定义一个函数，返回投资组合风险的平方
    def minimize_risk_for_target_return(weights):
        portfolio_return = np.sum(weights * mean_daily_returns)
        constraint = portfolio_return - possible_return
        return 10000 * constraint ** 2 + calculate_risk(weights, cov_matrix)  # 优化目标：最小化风险

    # 定义约束条件：权重之和等于1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    # 定义边界条件：每个权重在0和1之间
    bounds = tuple((0, 1) for asset in range(len(returns_columns)))
    # 使用SciPy的minimize函数来找到最小风险投资组合
    result = minimize(minimize_risk_for_target_return, len(returns_columns) * [1. / len(returns_columns), ], bounds=bounds, constraints=constraints)
    frontier_x.append(calculate_risk(result['x'], cov_matrix))

# 绘制有效前沿曲线
plt.plot(frontier_x, frontier_y, 'r--', linewidth=3, label='Efficient Frontier')
plt.legend(labelspacing=0.8)
plt.show()



