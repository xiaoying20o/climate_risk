#%%
import pandas as pd
import numpy as np
import re



#%%
finbert = pd.read_parquet("D:/data/finbert_acute.parquet", engine="fastparquet")
df = pd.read_parquet("D:/data/10X_permno.parquet")
print(finbert.shape)
print(df.shape)


# %%
# Befor we merge call and 10_k, caculate firm-level call finbert score to reduce the work time

def preview_except(df, exclude_cols=None, n=10):
    if exclude_cols is None:
        exclude_cols = []
    return df.drop(columns=exclude_cols).head(n)
#%%

# 
preview_except(finbert, ["componenttext", "altprc"], 20)

# %%

id_col   = "gvkey"   # use gvkey to merge
date_col = "date"
y_col    = "finbert_signed"


first_cols = ["companyid", "companyname", "altprc", "vol", "mktcap"]

# transfer date format
finbert_copy = finbert.copy()
finbert_copy[date_col] = pd.to_datetime(finbert_copy[date_col], errors="coerce")
finbert_copy = finbert_copy.dropna(subset=[id_col, date_col, y_col])

#  finbert score to numeric
finbert_copy[y_col] = pd.to_numeric(finbert_copy[y_col], errors="coerce")

# 
agg_dict = {y_col: "mean"}
for c in first_cols:
    if c in finbert_copy.columns:
        agg_dict[c] = "first"

# aggregate 
finbert_copy["year"] = finbert_copy[date_col].dt.year
g = finbert_copy.groupby([id_col, "year"])

panel_year = (
    g.agg(agg_dict)
     .reset_index()
     .rename(columns={y_col: "finbert_signed_mean"})
)

# also include observation number per year,per firm
size_df = g.size().reset_index(name="n_obs")
panel_year = panel_year.merge(size_df, on=[id_col, "year"], how="left")
front_cols = ["gvkey", "year", "companyid", "companyname",
              "altprc", "vol", "mktcap", "finbert_signed_mean", "n_obs"]
panel_year = panel_year[front_cols].sort_values(["gvkey", "year"]).reset_index(drop=True)



#%%
panel_year.head()

# %%
#same panel_finbert
panel_year.to_csv('panel_year_finbert.csv', index=False)



#%%
df10k[df10k['gvkey'] == 1982]

# %%
"""
然后我们从10-k中抽取panel 之前一个财政年的RF 和Mgm建立dummy variable 之后在与我们的panel data 合并，
这样来减少处理时间。

"""
#%%
# Keep only the relevant columns to avoid issues with duplicate merges.
# Dropping extra columns prevents unintended many-to-many merges.
df10k = df.loc[df['type'].str.startswith('10-K', na=False),
               ['date','gvkey','RF','Mgm']].copy()

df10k['date'] = pd.to_datetime(df10k['date'], errors='coerce')
df10k['file_year'] = df10k['date'].dt.year

# If a firm (gvkey) has multiple 10-K filings in the same year,
# keep only the first one (arbitrary but consistent)
df10k = df10k.drop_duplicates(['gvkey','file_year'], keep='first')
p = panel_year.rename(columns={'year':'call_year'})
p['lag_file_year'] = p['call_year'] - 1

# merge
subset_10k = p.merge(df10k,
                     left_on=['gvkey','lag_file_year'],
                     right_on=['gvkey','file_year'],
                     how='left')


# %%
# This block checks whether the 10-K merge worked as expected,
# to rule out NaNs caused by merge errors.
subset_10k["matched"] = subset_10k["file_year"].notna()

# Add a flag indicating whether a 10-K match was found
not_matched = subset_10k.loc[~subset_10k["matched"], ["gvkey", "call_year"]].drop_duplicates()

print("Number of unmatched samples:", len(not_matched))
print("First 10 examples:")
print(not_matched.head(10))

def check_10k_availability(gvkey, call_year, df10k):

    target_year = call_year - 1
    matched = df10k.loc[(df10k["gvkey"] == gvkey) & (df10k["file_year"] == target_year)]
    
    if matched.empty:
        print(f"❌ gvkey={gvkey}, call_year={call_year}, missing file_year={target_year} 10-K")
    else:
        print(f"✅ gvkey={gvkey}, call_year={call_year}, found file_year={target_year} 10-K")
        display(matched.head())

# Example checks
check_10k_availability(11976, 2014, df10k)
check_10k_availability(11976, 2015, df10k)


# %%
print(subset_10k.shape)
print(panel_year.shape)



"""
now we use RFS dictionary to creat dummy variable
"""

# %%
#-----------------clean text data of 10-k-------------------


def remove_toc_noise(text):
    lines = text.splitlines()

    # Filter out: only numbers, only "ITEM X.", title name (short word), or entire line capitalization
    cleaned_lines = []
    for line in lines:
        line = line.strip()

        if not line:  # skip 
            continue
        if re.match(r'^ITEM\s+\d+[A-Z]?\.*$', line, re.IGNORECASE):
            continue
        if re.match(r'^\d+$', line):  # all number line: pages' number
            continue
        if line.upper() == line and len(line) < 30:  # entire line is made of Capital(item name) 
            continue
        if line in ['Properties', 'Legal Proceedings', 'Unresolved Staff Comments']:
            continue

        cleaned_lines.append(line)

    return ' '.join(cleaned_lines)
def clean_linebreaks(text):
    # remove\n and spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)  # 
    return text.strip()
def remove_tabular_noise(text):
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        # if the whole line its just number,delete
        if re.match(r'^\d{4}$', line):  # year
            continue
        if re.match(r'^[\d,]+$', line):  # only number 
            continue

        cleaned.append(line)

    return ' '.join(cleaned)
def remove_urls(text):
    # ://xxx、www.xxx、
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)
#pipeline
def clean_text(raw_text):
    cleaned = remove_toc_noise(raw_text)
    cleaned = clean_linebreaks(cleaned)
    cleaned = remove_tabular_noise(cleaned)
    cleaned = remove_urls(cleaned)
    return cleaned

# %%
# Remove rows with NaNs:
# Some NaNs are due to no matching 10-K before the earnings call (no match found),
# but others are cases where a 10-K was matched, yet the relevant fields (e.g., RF or Mgm) are still missing.
# These should also be removed before generating dummy variables.

subset_10k = subset_10k[
    ~(
        subset_10k['Mgm'].isna() | (subset_10k['Mgm'].str.strip() == "") |
        subset_10k['RF'].isna() | (subset_10k['RF'].str.strip() == "")
    )
]

#%%
subset_10k['RF'] = subset_10k['RF'].apply(clean_text)
subset_10k['Mgm'] = subset_10k['Mgm'].apply(clean_text)
# %%
for idx, row in subset_10k.head(10).iterrows():
    print(f"\n=== Sample {idx} ===")
    print("\n--- RF (Risk Factors) ---")
    print(row['RF'])
    print("\n--- Mgm (MD&A) ---")
    print(row['Mgm'])
    print("="*80)


# %%
# load dictionary from RFS
file_path = 'D:/data/Climate Risk Dictionary_LSTY_2024.xlsx'
sheets_dict = pd.read_excel(file_path, sheet_name=None)

# check sheets' name
print(sheets_dict.keys())
# %%
Physical = sheets_dict['Acute Physical Risk']
Transition  = sheets_dict['Transition Risk']
Physical_keywords = set(Physical['Acute'].dropna().str.lower().str.strip())
Transition_keywords = set(Transition['Transition'].dropna().str.lower().str.strip())
# %%
def contains_phrase(text, phrase_set):
    text_lower = str(text).lower()
    return int(any(phrase in text_lower for phrase in phrase_set))
def add_risk_binaries(df, rf_col='RF', mgm_col='Mgm'):
    from tqdm import tqdm
    tqdm.pandas(desc="Checking RF keywords")

    df['RF_Physical'] = df[rf_col].progress_apply(lambda x: contains_phrase(x, Physical_keywords))
    df['RF_Transition'] = df[rf_col].progress_apply(lambda x: contains_phrase(x, Transition_keywords))

    tqdm.pandas(desc="Checking Mgm keywords")

    df['Mgm_Physical'] = df[mgm_col].progress_apply(lambda x: contains_phrase(x, Physical_keywords))
    df['Mgm_Transition'] = df[mgm_col].progress_apply(lambda x: contains_phrase(x, Transition_keywords))

    return df
#%%
subset_10k = add_risk_binaries(subset_10k, rf_col='RF', mgm_col='Mgm')

# %%
preview_except(subset_10k,["RF","Mgm"],10)
# %%
# # Select all columns related to disclosures
disclosure_cols = [
    'RF_Physical', 'RF_Transition', 
    'Mgm_Physical',  'Mgm_Transition'
]

# Create a new 'disclosure' column: set to 1 if any specified column has a value of 1, otherwise 0
subset_10k['disclosure'] = subset_10k[disclosure_cols].any(axis=1).astype(int)

#%%
# Select all columns related to RF-disclosures

RF_disclosure_cols = [
    'RF_Physical', 'RF_Transition'
]

# Create a new 'disclosure' column: set to 1 if any specified column has a value of 1, otherwise 0
subset_10k['RF_disclosure'] = subset_10k[RF_disclosure_cols].any(axis=1).astype(int)
# %%
# Select all columns related to RF-disclosures
Mgm_disclosure_cols = [
   'Mgm_Physical', 'Mgm_Transition'
]

# Create a new 'disclosure' column: set to 1 if any specified column has a value of 1, otherwise 0
subset_10k['Mgm_disclosure'] = subset_10k[Mgm_disclosure_cols].any(axis=1).astype(int)


#%%
preview_except(subset_10k,["RF","Mgm"],10)
# %%

subset_10k.to_parquet('subset_10k.parquet', index=False)
# %%
print(subset_10k.shape)
# %%

# %%
