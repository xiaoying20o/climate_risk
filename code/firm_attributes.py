#%%
import pandas as pd
#%%
#--------------------Extract Firms control variable--------------
import dask.dataframe as dd

df = dd.read_csv("C:/Users/xiaoying/Downloads/BK_char.csv", assume_missing=True)
print(list(df.columns))
#%%
df.head(10)
# %%
all_columns = df.columns.tolist()

# 你要查找的目标列（支持模糊匹配，大小写不敏感）
wanted_columns = ['ebit', 'debt', 'cash', 'capx', 'rd', 'assets', 'debt_at', 'ppen','RET','sic','ff49','permno','saleq']

# 
for col in wanted_columns:
    matches = [c for c in all_columns if col.lower() in c.lower()]
    if matches:
        print(f"✅ Found match(es) for '{col}': {matches}")
    else:
        print(f"❌ No match found for '{col}'")


# %%
data = pd.read_parquet("merged_call_quaterly.parquet")
# %%
df['gvkey'] = df['gvkey'].astype('Int64').astype(str) 
data['gvkey'] = data['gvkey'].astype(str)
target_gvkeys = data['gvkey'].unique()
df['date'] = dd.to_datetime(df['date'], errors='coerce')
df = df[df['date'].dt.year >= 2004]
df = df[df['gvkey'].isin(target_gvkeys)]
# %%
control_vars = ['assets', 'debt_at', 'cash_at', 'ppen_mev', 'ebit_at', 'capx_at', 'rd_at','ret','ff49','saleq_su','permno']
for col in control_vars:
    df[col] = dd.to_numeric(df[col], errors='coerce')

#%

#%%
from dask.diagnostics import ProgressBar
firm_attributes = df[['gvkey', 'date'] + control_vars]
with ProgressBar():
    firm_attributes_pd = firm_attributes.compute()

# %%
print(firm_attributes_pd.shape) 
#%%
print(data.shape) 
data.isna().sum()
#%%
firm_attributes_pd.head(20)



# %%
firm_attributes_pd.to_csv("firm_attributes2.csv", index=False)


# %%
