"""
在merge之后shape并不一样，是因为我们的finbert acute是基于提取出来的sentence level，对于一个
quater的call，可能提取出来不同的句子与actute 有关，所以我们现在聚合成季度的数据
"""
#%%
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