# %%
#-----------------------------Binary variable-------------------------
# load dictionary from RFS
file_path = 'D:/data/Climate Risk Dictionary_LSTY_2024.xlsx'
sheets_dict = pd.read_excel(file_path, sheet_name=None)
Physical = sheets_dict['Acute Physical Risk']
Transition  = sheets_dict['Transition Risk']
Physical_keywords = set(Physical['Acute'].dropna().str.lower().str.strip())
Transition_keywords = set(Transition['Transition'].dropna().str.lower().str.strip())
# %%
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
# %%
text_columns = ['prev_RF', 'prev_Mgm', 'next_Mgm', 'next_RF', ]

subset_df = add_dynamic_risk_binaries(
    full_df,
    text_columns=text_columns,
    physical_keywords=Physical_keywords,
    transition_keywords=Transition_keywords
)
