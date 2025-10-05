'''
This file is to calculate cosine similarity based on words 
'''
#%%
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
# Prepare dictionary
path1 = "D:/data/JF_dictionary/bigrams_07222021.pkl"    
path2 = "D:/data/JF_dictionary/physical_bigrams_4.pkl"   
path3 = "D:/data/JF_dictionary/opportunity_bigrams_4.pkl"  
path4 = "D:/data/JF_dictionary/regulatory_bigrams_4.pkl"  
total = pd.read_pickle(path1) 
physical = pd.read_pickle(path2)
opportunity= pd.read_pickle(path3)
regulatory =pd.read_pickle(path4)

# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter



      
# %%
#上有任务，先将所有的bigram抓出来
merged = pd.read_parquet("D:/projects/climate_risk/quarterly_panel_include_control_next10k.parquet")
print(merged.columns)



#%%

subset = merged[['call_date','gvkey','permno','prev_file_date','prev_file_type',
                 'prev_Mgm','next_10K_date','next_10K_Mgm']]






#%%
#--------#上有任务，先将所有的bigram抓出来-------------
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
def apply_hit_terms(df, col_name: str, dict_set: set, dict_name: str, show_progress=True) -> None:
    prefix = f"{col_name}_{dict_name}"
    mask = df[col_name].notna()
    docs = df.loc[mask, col_name].astype(str).tolist()

    per_doc = []
    iterator = tqdm(docs, desc=f"{col_name} × {dict_name}", unit="doc") if show_progress else docs
    for text in iterator:
        toks = _lemm_no_stop(text)
        hit_terms = []
        for a, b in bigrams(toks):
            phrase = f"{a} {b}"
            if phrase in dict_set:
                hit_terms.append(phrase)
        per_doc.append(hit_terms)

    
    df[f"{prefix}_hit_terms"] = pd.Series([[] for _ in range(len(df))], dtype="object")

    
    df.loc[mask, f"{prefix}_hit_terms"] = pd.Series(per_doc, index=df.loc[mask].index)
# %%
#------Lemma Dic--------
total_set = normalize_dict_bigrams(total)
physical_set = normalize_dict_bigrams(physical)
opportunity_set = normalize_dict_bigrams(opportunity)
regulatory_Set = normalize_dict_bigrams(regulatory)

#%%
cols = ['prev_Mgm','next_10K_Mgm']
dicts = {
    "physical": physical_set,
    "opportunity": opportunity_set,
   "regulatory": regulatory_Set,
   "Total": total_set
}
for col in cols:
    for dict_name, dict_set in dicts.items():
        print(f"→ Running: {col} × {dict_name}")
        apply_hit_terms(subset, col, dict_set, dict_name)

#%%%
subset.head()
#%%
subset.to_parquet('cosine_bigram.parquet',index=False)
# %%
#------------------Test Code--------------------------
data = {
    "prev_Mgm": [
        "These days temperature and flood frequency increase. Climate change makes drought and temperature issue severe."
    ],
    "next_Mgm": [
        "Temperature issue became serious as temperature keeps rising. Flood frequency has doubled in recent years."
    ]
}
df = pd.DataFrame(data)

climate_dict = {"temperature issue", "temperature increase", "flood frequency"}

apply_hit_terms(df, "prev_Mgm", climate_dict, "physical")
apply_hit_terms(df, "next_Mgm", climate_dict, "climate")



#%%
subset.isna().sum()






#%%
#---------------下游任务----------
def build_union(sentences):
    """
    构造 frequency-expanded union
    每个词展开成 [word#1, word#2, ..., word#max_count]
    """
    # count max 
    max_counts = Counter()
    for s in sentences:
        counts = Counter(s)
        for w, c in counts.items():
            max_counts[w] = max(max_counts[w], c)
    
    # expand to slots
    union = []
    for w, c in max_counts.items():
        for i in range(1, c+1):
            union.append(f"{w}#{i}")
    return union, max_counts

def vectorize(sentence, union, max_counts):
    """
    frequency-expanded vetor
    """
    vec = [0]*len(union)
    counts = Counter(sentence)
    for i, slot in enumerate(union):
        word, idx = slot.split("#")
        idx = int(idx)
        if counts[word] >= idx:
            vec[i] = 1
    return vec
#%%
def process_row(row, prev_columns, next_columns):
    result = {
        'gvkey': row['gvkey'],
        'call_date':row['call_date'],
        'permno':row['permno'],
        'prev_file_type':row['prev_file_type'],
        'prev_file_date':row['prev_file_date'],
        'next_10K_date':row['next_10K_date']
    }

    for prev_col, next_col in zip(prev_columns, next_columns):

        prev = row[prev_col]
        next_ = row[next_col]

        
        prev = prev if isinstance(prev, list) else []
        next_ = next_ if isinstance(next_, list) else []

        union, max_counts = build_union([prev, next_])
        vec_prev = vectorize(prev, union, max_counts)
        vec_next = vectorize(next_, union, max_counts)
        if sum(vec_prev) == 0 and sum(vec_next) == 0:
            cosine = 0.0
        else:
            cosine = cosine_similarity([vec_prev], [vec_next])[0][0]
        topic = prev_col.replace('prev_Mgm_', '').replace('_hit_terms', '')
        result[f'cosine_{topic}'] = cosine
        result[f'union_len_{topic}'] = len(union)
        result[f'vec_prev_{topic}'] = vec_prev
        result[f'vec_next_{topic}'] = vec_next

    return pd.Series(result)

# %%
#------------Test code
from sklearn.metrics.pairwise import cosine_similarity

prev_cols = [
    'prev_Mgm_opportunity_hit_terms',
    'prev_Mgm_physical_hit_terms',
    'prev_Mgm_regulatory_hit_terms',
    'prev_Mgm_Total_hit_terms'
]

next_cols = [
    'next_10K_Mgm_opportunity_hit_terms',
    'next_10K_Mgm_physical_hit_terms',
    'next_10K_Mgm_regulatory_hit_terms',
    'next_10K_Mgm_Total_hit_terms'
]
test_subset = subset.head(3).copy()
test_result = test_subset.apply(lambda row: process_row(row, prev_cols, next_cols), axis=1)


# %%
# check test
print(test_result)
for i, row in enumerate(test_subset['next_10K_Mgm_opportunity_hit_terms']):
    print(f"Row {i}: {row}")

# %%
cosine_result = subset.apply(lambda row: process_row(row, prev_cols, next_cols), axis=1)

#%%
cosine_result.head()
# %%
cosine_result.shape
# %%
cosine_result.to_parquet("cosine_result.parquet", index=False)
# %%
cosine = pd.read_parquet("D:/projects/climate_risk/cosine_result.parquet")
# %%
cosine.head()
# %%
