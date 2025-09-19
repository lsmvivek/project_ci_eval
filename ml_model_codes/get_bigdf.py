import pandas as pd
import glob

def get_bigdf(target_, dir_, dates, save_path=None):
    train_start_date, train_end_date, test_start_date, test_end_date = dates
    print(f'\n{"="*60}')
    print(f"Compiling data from {dir_} for target {target_}...")
    print(f"Train dates: {train_start_date} to {train_end_date}")
    print(f"Test dates: {test_start_date} to {test_end_date}")
    # Gather all variable file paths
    path_all_vars = sorted(glob.glob(f"/storage/extracted_grids/{dir_}/*.txt"))

    # Read and concatenate all variable data
    dfs = [pd.read_csv(path, header=0, index_col=0) for path in path_all_vars]
    big_df = pd.concat(dfs, axis=1)

    # Drop excluded variables
    excluded_vars = [
        'SnowT_tavg', 'Swe_tavg', 'SnowDepth_tavg', 'Snowf_tavg',
        'EvapSnow_tavg', 'Qsm_tavg'
    ]
    big_df = big_df.drop(columns=[col for col in excluded_vars if col in big_df.columns])

    # Create contemporaneous time series data
    big_df_contemp_ts = big_df.shift(-1).dropna(inplace=False)
    big_df_contemp_ts.index = big_df.index[1:]  # Adjust index to match the lagged data
    # Create lagged time series data
    big_df_lagged_ts = big_df.shift(1).dropna(inplace=False)
    big_df_lagged_ts.index = big_df.index[1:]  # Adjust index to match the lagged data
    big_df_lagged_ts.columns = [f"{col}_lag1" for col in big_df.columns]

    # Concatenate contemporaneous and lagged data
    result_df = pd.concat([big_df_contemp_ts, big_df_lagged_ts], axis=1)
    result_df.index = pd.to_datetime(result_df.index)
    if save_path:
        result_df.to_csv(save_path, index=True, header=True, sep='\t')
        print(f'Saved big_df in {save_path}')
        print(f'\n{"="*60}')
    return result_df