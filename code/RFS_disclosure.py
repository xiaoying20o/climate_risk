

# %%
import numpy as np
import pandas as pd

#%%
#%%
# RFS disclosure
RFS_disclosure_after = pd.read_csv(r"D:\data\RFS_after_2019.csv")
RFS_disclosure_before = pd.read_csv(r"D:\data\RFS_before_2019.csv")
# %%
print(RFS_disclosure_after.head())
# %%
print(RFS_disclosure_before.head())
# %%
RFS_disclosure_all = pd.concat(
    [RFS_disclosure_before, RFS_disclosure_after],
    axis=0,              # 按行拼接
    ignore_index=True    # 重新生成连续的索引
)

# 检查结果
print(RFS_disclosure_all.shape)
print(RFS_disclosure_all.head())
print(RFS_disclosure_all.tail())
# %%
RFS_disclosure_all.to_csv("RFS_disclosure_all.csv", index=False)
# %%
