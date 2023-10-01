import pandas as pd
from scipy.interpolate import UnivariateSpline

def economic_indicator_analyze():
    # Load the Excel file
    filename = 'Economic_Indicators.xlsx'
    xls = pd.ExcelFile(filename)

    # Load the sheet into a DataFrame
    df = xls.parse('Sheet1')

    # Define the indices to extract
    indices_to_extract = [
        "Unemployment Rate (sa)",
        "CPI, TD-MI Inflation Gauge Idx (%m/m)",
        "Money Supply, M1 (%y/y)",
        "Money Supply, M3 (%y/y)",
        "CPI (%q/q)",
        "PPI (%q/q)",
        "Real GDP Growth (%q/q, sa)"
    ]

    # Define the columns to extract corresponding to Feb and Mar 2020
    cols_to_extract = ['Key Indicators - Australia', 'Unnamed: 7', 'Unnamed: 6']

    # Extract the required data
    extracted_df = df.loc[df['Key Indicators - Australia'].isin(indices_to_extract), cols_to_extract]

    # Rename the columns
    extracted_df.columns = ['Indicator', '2020-02', '2020-03']

    # Melt the dataframe to sort by date
    extracted_df = extracted_df.melt(id_vars=['Indicator'], var_name='Date', value_name='Value')

    # Pivot the original extracted DataFrame to get the desired format
    pivot_df = extracted_df.pivot(index='Date', columns='Indicator', values='Value')

    # Insert additional data points
    pivot_df.loc['2020-02-15'] = [0.51, -0.084, 24.58, 2.17, 0.444, 0.1657, 5.0834]
    pivot_df.loc['2020-02-16'] = [0.55, -0.074, 24.40, 2.19, 0.504, 0.166, 4.0834]
    pivot_df = pivot_df.sort_index()  # Sort the DataFrame by index after insertion

    # Convert date indices to numerical values (number of days from the start) for the pivot_df DataFrame
    numerical_index = (pd.to_datetime(pivot_df.index) - pd.Timestamp('2020-02-01')).days
    pivot_df_numerical_index = pivot_df.copy()
    pivot_df_numerical_index.index = numerical_index

    # Create a DataFrame with consistent date intervals for interpolation
    all_dates = pd.date_range(start='2020-02-01', end='2020-03-01', periods=42).strftime('%Y-%m-%d')
    all_index = (pd.to_datetime(all_dates) - pd.Timestamp('2020-02-01')).days
    spline_interpolated_df_numerical_index = pd.DataFrame(index=all_index, columns=pivot_df.columns)

    # Loop over each column to perform quadratic spline interpolation
    for col in spline_interpolated_df_numerical_index.columns:
        # Extract known (non-NaN) values
        known_data = pivot_df_numerical_index[col].dropna()
        known_index = known_data.index
        known_values = known_data.values.astype(float)

        # Now we should have more than 2 points for quadratic spline interpolation
        if len(known_values) >= 3:  # We need at least 3 points for quadratic spline interpolation
            # Construct quadratic spline interpolation function
            spl = UnivariateSpline(known_index, known_values, k=2, s=0)  # k=2 for quadratic spline

            # Interpolate
            interpolated_values = spl(all_index)

            # Assign the interpolated values back to the DataFrame
            spline_interpolated_df_numerical_index[col] = interpolated_values

    # Convert numerical index back to date strings
    spline_interpolated_df = spline_interpolated_df_numerical_index.copy()
    spline_interpolated_df.index = all_dates
    # print(spline_interpolated_df)
    return spline_interpolated_df
    # print(spline_interpolated_df,type(spline_interpolated_df))
    # Convert the interpolated DataFrame to a numpy array, if needed
    # new_numpy_array_spline_interpolated = spline_interpolated_df.values
    # print(new_numpy_array_spline_interpolated,type(new_numpy_array_spline_interpolated))

economic_indicator_analyze()