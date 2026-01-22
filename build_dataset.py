import pandas as pd
from data.preprocessing import merge_sources

# load raw data 
df_ft = pd.read_parquet("data/ft_raw.parquet")
df_wttj = pd.read_parquet("data/wttj_raw.parquet")

# preprocess
df_jobs = merge_sources(df_wttj=df_wttj, df_ft=df_ft)

# save
df_jobs.to_parquet("data/jobs_preprocessed.parquet")

print("Dataset sauvegardé")
