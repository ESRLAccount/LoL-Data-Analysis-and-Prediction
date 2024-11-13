import pandas as pd

file_path= "../data/OutputRank/ReadyToPlot/masterfile_an.csv"
file_path2= "../data/OutputRank/ReadyToPlot/masterfile_an2.csv"

df = pd.read_csv(file_path)

# Create pivot table
pivot_df = pd.pivot_table(df,
                          index='PID',
                          columns='COND',
                          values=['RMS_iMVC_PVT_FDS_avg', 'RMS_iMVC_PVT_ED_avg', 'RMS_iMVC_PVT_FDI_avg'],
                          aggfunc='first')


pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]

# Reset index to make 'PID' a column
pivot_df = pivot_df.reset_index()
pivot_df.to_csv(file_path2)
print(pivot_df)