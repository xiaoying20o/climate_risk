#%%
"""
This file is for calculate 10x context based on JF dictionary.
"""
import pandas as pd
from itertools import tee
from collections import Counter
from tqdm.auto import tqdm
import re
import numpy as np
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from joblib import Parallel, delayed

#%%
path1 = "D:/data/JF_dictionary/bigrams_07222021.pkl"    
path2 = "D:/data/JF_dictionary/physical_bigrams_4.pkl"   
path3 = "D:/data/JF_dictionary/opportunity_bigrams_4.pkl"  
path4 = "D:/data/JF_dictionary/regulatory_bigrams_4.pkl"  

# %%
total = pd.read_pickle(path1) 
physical = pd.read_pickle(path2)
opportunity= pd.read_pickle(path3)
regulatory =pd.read_pickle(path4)

#%%
print(opportunity)
#%%
df = pd.read_parquet("quarterly_panel_include_control.parquet")



#%%
#-------Merge Next 10-k to see if it is good for our feedback test----
file = pd.read_parquet("D:/projects/climate_risk/10x_clean.parquet")
file.head()

#%%
file['type'].unique()
#%%
# 定义年报集合
k_types = {"10-K", "10-K405", "10KSB"}

# 筛选出所有年报
df_10k = file[file["type"].isin(k_types)].copy()
df_10k = df_10k[["gvkey", "date", "RF", "Mgm"]].copy()
# 3. 按 gvkey 和日期排序
df_10k = df_10k.sort_values(["date"])
df_calls = df.sort_values(["call_date"])
df_calls = df_calls.rename(columns={"gvkey_x": "gvkey"})
df_10k["gvkey"] = df_10k["gvkey"].astype(str)
df_calls["gvkey"] = df_calls["gvkey"].astype(str)
# 4. merge_asof: 找到 call_date 之后的第一个 10-K
merged = pd.merge_asof(
    df_calls,
    df_10k,
    left_on="call_date",   # earnings call 时间
    right_on="date",       # 10-K 文件发布日期
    by="gvkey",            # 按公司匹配（注意 gvkey/gvkey_x 命名统一）
    direction="forward",    # 找 call_date 之后最近的 10-K
    allow_exact_matches=False
)

#%%
merged = merged.rename(columns={
    "RF": "next_10K_RF",
    "Mgm": "next_10K_Mgm",
    "date": "next_10K_date"
})

merged.to_parquet("quarterly_panel_include_control_next10k.parquet", index=False)


#%%
merged = pd.read_parquet("quarterly_panel_include_control_next10k.parquet")
#%%
#  lemma / stopwords
lemm = WordNetLemmatizer()

STOP = set(stopwords.words("english")) - {"no", "not", "nor"} 

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-\.][a-z0-9]+)*", re.I) 
def _lemm_no_stop(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text.lower())
    lemmas = [lemm.lemmatize(tok) for tok in toks if tok not in STOP]
    return lemmas

def bigrams(tokens):
    a, b = tee(tokens)  # 2 iterator
    next(b, None)    
    return zip(a, b)  # return bigram

def normalize_dict_bigrams(phrases: list[str]) -> set[str]:
    return {
        " ".join(lemm.lemmatize(tok) for tok in phrase.lower().split() if tok not in STOP)
        for phrase in phrases
    }


def top_bigrams(texts: list[str], n=100):
    counter = Counter()
    for text in texts:
        toks = _lemm_no_stop(text)
        for a, b in bigrams(toks):
            counter[f"{a} {b}"] += 1
    return counter.most_common(n)

# %%
GLOBAL_BIGRAM_COUNTER = Counter()


def apply_hit_rate(df, col_name: str, dict_set: set, dict_name: str, collect_bigrams=True, show_progress=True) -> None:
    prefix = f"{col_name}_{dict_name}"
    mask = df[col_name].notna()
    docs = df.loc[mask, col_name].astype(str).tolist()

    per_doc = []
    iterator = tqdm(docs, desc=f"{col_name} × {dict_name}", unit="doc") if show_progress else docs
    for text in iterator:
        toks = _lemm_no_stop(text)
        B = max(len(toks) - 1, 0)
        h, hit_terms = 0, []
        for a, b in bigrams(toks):
            phrase = f"{a} {b}"
            if phrase in dict_set:
                h += 1
                hit_terms.append(phrase)

        if collect_bigrams:
            GLOBAL_BIGRAM_COUNTER.update(hit_terms)

        per_doc.append({"hits": h, "bigrams": B})

    # --- 1. hit_count (依赖字典)
    df[f"{prefix}_hit_count"] = np.nan
    df.loc[mask, f"{prefix}_hit_count"] = [d["hits"] for d in per_doc]

    # --- 2. bigram_len (不依赖字典，只生成一次)
    len_col = f"{col_name}_len"
    if len_col not in df.columns:  # 避免重复覆盖
        df[len_col] = np.nan
        df.loc[mask, len_col] = [d["bigrams"] for d in per_doc]

    # --- 3. hit_rate (依赖字典)
    rate_col = f"{prefix}_hit_rate"
    df[rate_col] = np.nan
    df.loc[mask & (df[len_col] > 0), rate_col] = (
        df.loc[mask, f"{prefix}_hit_count"] / df.loc[mask, len_col]
    )
#%%
#------Lemma Dic--------
total_set = normalize_dict_bigrams(total)
physical_set = normalize_dict_bigrams(physical)
opportunity_set = normalize_dict_bigrams(opportunity)
regulatory_Set = normalize_dict_bigrams(regulatory)

#%%
dicts = {
    "physical": physical_set,
    "opportunity": opportunity_set,
   "regulatory": regulatory_Set,
   "Total": total_set
}
#%%
GLOBAL_BIGRAM_COUNTER.clear()
cols = ["prev_RF","prev_Mgm", "next_Mgm", "next_RF",'next_10K_RF','next_10K_Mgm']

for col in cols:
    for dict_name, dict_set in dicts.items():
        print(f"→ Running: {col} × {dict_name}")
        apply_hit_rate(merged, col, dict_set, dict_name)


top100 = GLOBAL_BIGRAM_COUNTER.most_common(100)
top100_df = pd.DataFrame(top100, columns=["bigram", "count"])

top100_df.to_csv("top100_bigrams.csv", index=False)
JF_score_10x_lemma = merged[['call_date','gvkey','permno',
       'prev_RF_physical_hit_count','prev_RF_opportunity_hit_count', 'prev_RF_regulatory_hit_count','prev_RF_Total_hit_count',
       'prev_RF_physical_hit_rate','prev_RF_opportunity_hit_rate','prev_RF_regulatory_hit_rate','prev_RF_Total_hit_rate','prev_RF_len',
        'prev_Mgm_physical_hit_count','prev_Mgm_opportunity_hit_count', 'prev_Mgm_regulatory_hit_count', 'prev_Mgm_Total_hit_count', 
        'prev_Mgm_physical_hit_rate','prev_Mgm_opportunity_hit_rate','prev_Mgm_regulatory_hit_rate','prev_Mgm_Total_hit_rate','prev_Mgm_len',
        'next_Mgm_physical_hit_count','next_Mgm_opportunity_hit_count', 'next_Mgm_regulatory_hit_count', 'next_Mgm_Total_hit_count',
        'next_Mgm_physical_hit_rate','next_Mgm_opportunity_hit_rate', 'next_Mgm_regulatory_hit_rate', 'next_Mgm_Total_hit_rate','next_Mgm_len',
        'next_RF_physical_hit_count','next_RF_opportunity_hit_count', 'next_RF_regulatory_hit_count','next_RF_Total_hit_count',
        'next_RF_physical_hit_rate','next_RF_opportunity_hit_rate', 'next_RF_regulatory_hit_rate','next_RF_Total_hit_rate','next_RF_len',
        'next_10K_Mgm_physical_hit_count','next_10K_Mgm_opportunity_hit_count', 'next_10K_Mgm_regulatory_hit_count', 'next_10K_Mgm_Total_hit_count',
        'next_10K_Mgm_physical_hit_rate','next_10K_Mgm_opportunity_hit_rate', 'next_10K_Mgm_regulatory_hit_rate', 'next_10K_Mgm_Total_hit_rate','next_10K_Mgm_len',
        'next_10K_RF_physical_hit_count','next_10K_RF_opportunity_hit_count', 'next_10K_RF_regulatory_hit_count','next_10K_RF_Total_hit_count',
        'next_10K_RF_physical_hit_rate','next_10K_RF_opportunity_hit_rate', 'next_10K_RF_regulatory_hit_rate','next_10K_RF_Total_hit_rate','next_10K_RF_len'
          ]]

JF_score_10x_lemma.to_parquet('JF_score_10x_lemma.parquet', index=False)


#%%
GLOBAL_BIGRAM_COUNTER
#%%
baseline2=pd.read_parquet('C:/Users/xiaoying/Downloads/merged2_leamma_last_week.parquet')

#%%
baseline2['gvkey'] = baseline2['gvkey'].astype(str)
JF_score_10x_lemma['gvkey'] = JF_score_10x_lemma['gvkey'].astype(str)
baseline2['call_date'] = pd.to_datetime(baseline2['call_date'], errors='coerce') 
JF_score_10x_lemma['call_date'] = pd.to_datetime(JF_score_10x_lemma['call_date'], errors='coerce')
#%%
df_check = JF_score_10x_lemma[['gvkey', 'call_date', 'prev_Mgm_Total_hit_count']].merge( 
    baseline2[['gvkey', 'call_date', 'prev_Mgm_Total_hit_count']], 
    on=['gvkey', 'call_date'], suffixes=('_JF', '_baseline') )


#%%
df_check[
    df_check['prev_Mgm_Total_hit_count_JF'].notna() &
    df_check['prev_Mgm_Total_hit_count_baseline'].notna() &
    (df_check['prev_Mgm_Total_hit_count_JF'] != df_check['prev_Mgm_Total_hit_count_baseline'])
]




# %%
#-----Test  Sample------

df_test= pd.DataFrame({
    "prev_RF": [
        "Storm water management is crucial in coastal areas storm water",     # ✅ 命中 "storm water","coastal areas"
        "The sea level has risen due to global warm trends",      # ✅ 命中 "sea level", "global warm"
        "Unrelated sentence about finance and marketing",         # ❌ 无命中
        None                                                    # ❌ 测试 NaN                                              
    ],
    "prev_Mgm":[
        "",                                                       # ❌ 空字符串
        "Nickel metal and fluorine product are dangerous",        # ✅ 两个 hit
        "We expect a strong preseason due to heavy snow",         # ✅ 两个 hit
        "coastal region development continues"                    # ✅ 命中             
    ],
    "next_RF": [
        "Storm water management is crucial in coastal areas storm water",     # ✅ 命中 "storm water","coastal areas"
        "The sea level has risen due to global warm trends",      # ✅ 命中 "sea level", "global warm"
        "Unrelated sentence about finance and marketing",         # ❌ 无命中
        None                                                     # ❌ 测试 NaN                                              
    ],
    "next_Mgm":[
        "",                                                       # ❌ 空字符串
        "Nickel metal and fluorine product are dangerous",        # ✅ 两个 hit
        "We expect a strong preseason due to heavy snow",         # ✅ 两个 hit
        "coastal region development continues"                    # ✅ 命中             
    ],
    "next_10K_RF": [
        "Storm water management is crucial in coastal areas storm water",     # ✅ 命中 "storm water","coastal areas"
        "The sea level has risen due to global warm trends",      # ✅ 命中 "sea level", "global warm"
        "Unrelated sentence about finance and marketing",         # ❌ 无命中
        None                                                     # ❌ 测试 NaN                                              
    ],
    "next_10K_Mgm":[
        "",                                                       # ❌ 空字符串
        "Nickel metal and fluorine product are dangerous",        # ✅ 两个 hit
        "We expect a strong preseason due to heavy snow",         # ✅ 两个 hit
        "coastal region development continues"                    # ✅ 命中             
    ]
})
cols = ["prev_RF", "prev_Mgm", "next_Mgm", "next_RF", "next_10K_RF", "next_10K_Mgm"]
for col in cols:
    for dict_name, dict_set in dicts.items():
        print(f"→ Running: {col} × {dict_name}")
        apply_hit_rate(df_test, col, dict_set, dict_name)










# %%
