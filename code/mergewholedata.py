#%%
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
import os
# %%
#------------------Load Data---------------
df = pd.read_parquet("D:/projects/climate_risk/10x_clean.parquet")
print(df.shape)
call = pd.read_parquet("D:/data/transcript_permno.parquet")
print(call.shape)
# %%
#-----------------Preview-------------------
def preview_except(df, exclude_cols=None, n=10):
    if exclude_cols is None:
        exclude_cols = []
    return df.drop(columns=exclude_cols).head(n)
#%%
preview_except(call, ["componenttext"], 20)
#%%
preview_except(df,["Mgm","RF"],20)


#%%
# 确保 date 列是 datetime 类型
df['date'] = pd.to_datetime(df['date'])

# 查看最早和最晚的日期
start_date = df['date'].min()
end_date = df['date'].max()
#%%
print(start_date)
print(end_date)


#10k和call都是到2023年


#%%
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
# 只取 2003 年 1 月 1 日之后的数据
filings = df[df['date'] >= pd.Timestamp('2003-01-01')].copy()

#%%
calls = call[call['earnings_type'] == 1].copy()

#%%

calls['date'] = pd.to_datetime(calls['date'])
calls['anchor_date'] = calls['date'] + pd.DateOffset(months=2)
calls = calls.sort_values(['date']) 
filings = filings.sort_values(['date']) 









#%%
WIN = 365

# pre filing
prev = pd.merge_asof(
    left=calls,
    right=filings[['gvkey', 'date', 'Mgm', 'RF','type']].rename(columns={
        'Mgm': 'prev_Mgm',
        'RF':  'prev_RF',
        'date': 'prev_file_date',
        'type': 'prev_file_type'
    }),
    by='gvkey',
    left_on='date',
    right_on='prev_file_date',
    direction='backward',
    tolerance=pd.Timedelta(days=WIN),
    allow_exact_matches=False
)

# next filing
out = pd.merge_asof(
    left=prev.sort_values(['anchor_date'], kind='mergesort'),
    right=filings[['gvkey', 'date', 'Mgm', 'RF','type']].rename(columns={
        'Mgm': 'next_Mgm',
        'RF':  'next_RF',
        'date': 'next_file_date',
        'type': 'next_file_type'
    }),
    by='gvkey',
    left_on='anchor_date',
    right_on='next_file_date',
    direction='forward',
    tolerance=pd.Timedelta(days=WIN),
    allow_exact_matches=False
)

# rename 'date' : 'call_date'
out = out.rename(columns={'date': 'call_date'})
out.head(10)


#%%
print(out.columns)
#%%
preview_except(out,['month','companyid','companyname','transcriptid','headline', 'earnings_type',
                    'componenttext', 'altprc',
                    'vol', 'mktcap', 'anchor_date'],50)

#%%
out.isna().sum()
#%%
calls.shape
# %%
#--------------------#clean context for call context------------------------

def clean_componenttext(text):
        # delate [ph]、[Operator Instructions]
    text = re.sub(r"\[.*?\]", "", text)

    # 2. delate URL（http or www ）
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. eamil
    text = re.sub(r"\S+@\S+", "", text)

    # 4. space
    text = re.sub(r"\s+", " ", text).strip()

    return text

#clean context for the whole earning call
out["componenttext"] = out["componenttext"].apply(clean_componenttext)
















# %%
import os

from tqdm import tqdm

def save_pickle_in_chunks(df, base_filename, chunk_size=10000):
    os.makedirs("pkl_chunks", exist_ok=True)
    for i in tqdm(range(0, len(df), chunk_size)):
        chunk = df.iloc[i:i+chunk_size]
        filename = f"pkl_chunks/{base_filename}_part{i//chunk_size}.pkl"
        chunk.to_pickle(filename)
# %%
save_pickle_in_chunks(out, "merge_clean10x")


#%%
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

def load_pickle_chunks(directory, base_filename):
    all_chunks = sorted(glob(f"{directory}/{base_filename}_part*.pkl"))
    dfs = []

    for file in tqdm(all_chunks, desc="📦 Loading pkl chunks"):
        df = pd.read_pickle(file)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    print("✅ 所有分块已合并完成！")
    return merged_df
full_df = load_pickle_chunks("pkl_chunks", "merge_clean10x")

#%%
full_df.head()


# %%
#-----------------------------Binary variable-------------------------
# load dictionary from RFS
file_path = 'D:/data/Climate Risk Dictionary_LSTY_2024.xlsx'
sheets_dict = pd.read_excel(file_path, sheet_name=None)
Physical = sheets_dict['Acute Physical Risk']
Transition  = sheets_dict['Transition Risk']
Physical_keywords = set(Physical['Acute'].dropna().str.lower().str.strip())
Transition_keywords = set(Transition['Transition'].dropna().str.lower().str.strip())
def contains_phrase_safe(text, phrase_set):
    if pd.isna(text):
        return np.nan
    text_lower = str(text).lower()
    return int(any(phrase in text_lower for phrase in phrase_set))
def add_dynamic_risk_binaries(df, text_columns, physical_keywords, transition_keywords):
    from tqdm import tqdm
    tqdm.pandas()

    for col in text_columns:
        tqdm.pandas(desc=f"Checking {col} for Physical")
        df[f'{col}_Physical'] = df[col].progress_apply(lambda x: contains_phrase_safe(x, physical_keywords))

        tqdm.pandas(desc=f"Checking {col} for Transition")
        df[f'{col}_Transition'] = df[col].progress_apply(lambda x: contains_phrase_safe(x, transition_keywords))

    return df
#%%
#--------------------Run Binary
text_columns = ['prev_RF', 'prev_Mgm', 'next_Mgm', 'next_RF', ]

subset_df = add_dynamic_risk_binaries(
    full_df,
    text_columns=text_columns,
    physical_keywords=Physical_keywords,
    transition_keywords=Transition_keywords
)
#%%
preview_except(subset_df,['month','companyid','companyname','transcriptid','headline', 'earnings_type',
                    'componenttext', 'altprc',
                    'vol', 'mktcap', 'anchor_date'],20)
# %%
print(subset_df.columns.tolist())

#%%#--------处理空字符串

def sync_main_and_dummies(df, main_col, dummy_cols):
    df[main_col] = (
        df[main_col].astype(str).str.strip()
          .replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan})
    )
    mask_na = df[main_col].isna()
    df.loc[mask_na, dummy_cols] = np.nan

# 用法：
sync_main_and_dummies(subset_df, 'prev_Mgm', ['prev_Mgm_Physical', 'prev_Mgm_Transition'])
sync_main_and_dummies(subset_df, 'prev_RF',  ['prev_RF_Physical',  'prev_RF_Transition'])
# 需要的话继续：
sync_main_and_dummies(subset_df, 'next_Mgm', ['next_Mgm_Physical', 'next_Mgm_Transition'])
sync_main_and_dummies(subset_df, 'next_RF',  ['next_RF_Physical',  'next_RF_Transition'])









# %%
#————————————————This is for merge acute context---------------
finbert_acute = pd.read_parquet("D:/data/finbert_acute.parquet")
finbert_acute.head()
# %%
print(finbert_acute.columns.tolist())

#%%
"""
在merge之后shape并不一样，是因为我们的finbert acute是基于提取出来的sentence level，对于一个
quater的call，可能提取出来不同的句子与actute 有关，所以我们现在聚合成季度的数据
"""
# ==== 预处理 ====
df = finbert_acute.copy()
df["month"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["gvkey", "month", "finbert_signed"])
df["year"] = df["month"].dt.year
df["quarter"] = df["month"].dt.quarter

# ==== 聚合方式 ====
agg_dict = {
    "finbert_signed": "mean",
    "companyid": "first",
    "permno": "first",
    "companyname": "first",
    "transcriptid": "first",
    "headline": "first",
    "earnings_type": "first",
    "componenttext": "first",
    "altprc": "first",
    "vol": "first",
    "mktcap": "first",
    "date": "first"
}

# ==== 聚合 ====
panel_quarter = (
    df.groupby(["gvkey", "year", "quarter"])
      .agg(agg_dict)
      .reset_index()
      .rename(columns={"finbert_signed": "finbert_signed_mean"})
)

# ==== 观测数 ====
size_df = df.groupby(["gvkey", "year", "quarter"]).size().reset_index(name="n_obs")
panel_quarter = panel_quarter.merge(size_df, on=["gvkey", "year", "quarter"], how="left")

# ==== 列顺序 ====
cols_order = ["gvkey", "date", "finbert_signed_mean", "n_obs"]

panel_quarter = panel_quarter[cols_order]




# %%
# 设定要保留的列（来自 finbert_acute 的）
keep_cols = ['finbert_signed_mean','n_obs']  # 替换为你需要的列名

# 先确保日期列格式统一
subset_df['call_date'] = pd.to_datetime(subset_df['call_date'])
panel_quarter['date'] = pd.to_datetime(panel_quarter['date'])

merged_df1 = pd.merge(
    subset_df,
    panel_quarter[['date', 'gvkey'] + keep_cols],
    left_on=['call_date', 'gvkey'],
    right_on=['date', 'gvkey'],
    how='left'
)

# %%
print(subset_df.shape)
# %%merged_df1
print(merged_df1.shape)
#%%
merged_df1.to_parquet("merged_call_quaterly.parquet", index=False)
#%%
preview_except(merged_df1,['month','companyid','companyname','transcriptid','headline', 'earnings_type',
                    'componenttext', 'altprc',
                    'vol', 'mktcap', 'anchor_date'],50)
#%%
import pyarrow.parquet as pq

md = pq.read_metadata("merged_call_quaterly.parquet")
print("rows:", md.num_rows)
print("row groups:", md.num_row_groups)
for i in range(md.num_row_groups):
    rg = md.row_group(i)
    print(f"row_group {i}: rows={rg.num_rows}")




# %%
data.head(10)
# %%
print(data.shape)
# %%
