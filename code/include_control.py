#%%
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

#%%
data = pd.read_parquet("merged_call_quaterly.parquet")
# %%
#-----------Only select dataset which contains finbert number---------
subset_climate = data[data['finbert_signed_mean'].notna()]
# %%
subset_climate.head()
#%%
subset_climate.isna().sum()



# %%
#----------------------Combine firm attribute---------------

control2 =  pd.read_csv("firm_attributes2.csv")
#%%
#  permno to Int64（
control2['permno'] = pd.to_numeric(control2['permno'], errors='coerce').astype('Int64')

#  subset_climate-permno to Int64（从 int32 → Int64）
subset_climate['permno'] = subset_climate['permno'].astype('Int64')
#%%
subset_climate['month'] = (subset_climate['call_date'] - pd.offsets.MonthEnd(1)).dt.to_period("M")
control2['date'] = pd.to_datetime(control2['date'])

# merge 
control2['month'] = control2['date'].dt.to_period("M")
merged = pd.merge(
    subset_climate,
    control2,
    on=['permno', 'month'],
    how='left'
)

# %%
merged.isna().sum()
# %%
merged.sample(n=10, random_state=42)
#%%
print(merged.columns)



# %%
merged.to_parquet("quarterly_panel_include_control.parquet", index=False)
# %%
