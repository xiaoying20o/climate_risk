#%%
import pandas as pd
import re
from tqdm import tqdm
from typing import List, Tuple, Iterable, Dict, Any
from nltk.stem import WordNetLemmatizer
from blingfire import text_to_sentences
from typing import List, Tuple, Iterable, Dict, Any
#%%
call_clean = pd.read_parquet("D:/data/earnings_cleaned.parquet")
print(call_clean.shape)
call_clean.head()



# %%
#%%
#----------Load JF dictionary ------------
path1 = "D:/data/JF_dictionary/bigrams_07222021.pkl"    
path2 = "D:/data/JF_dictionary/physical_bigrams_4.pkl"   
path3 = "D:/data/JF_dictionary/opportunity_bigrams_4.pkl"  
path4 = "D:/data/JF_dictionary/regulatory_bigrams_4.pkl"  

total = pd.read_pickle(path1) 
physical = pd.read_pickle(path2)
opportunity= pd.read_pickle(path3)
regulatory =pd.read_pickle(path4)

#%%
print("total:", type(total))
print("physical:", type(physical))
print("opportunity:", type(opportunity))
print("regulatory:", type(regulatory))



#%%
# 拼接成一个大列表
all_keywords = total + physical + opportunity + regulatory

# 去重
all_keywords = list(set(all_keywords))

print("合并后的关键词总数:", len(all_keywords))
print(all_keywords[:20])  # 查看前20个
#%%
#---------Load RFS Dictionary---------
# keywords from RFS dictionary
# load dictionary from RFS
path = r"D:/data/Climate Risk Dictionary_LSTY_2024.xlsx"
Acute = pd.read_excel(path, sheet_name="Acute Physical Risk", engine="openpyxl")

print(Acute.columns)  
print(Acute.head())
Acute_keywords = Acute['Acute'].str.lower().tolist()
print(type(Acute_keywords))



#%%
lemm = WordNetLemmatizer()
# ---- Dictionary Lemmatization ----
def lemmatize_list(words: Iterable[str]) -> List[str]:
    out = []
    for w in words:
        tokens = re.findall(r"\b\w+\b", str(w).lower())
        lemmas = [lemm.lemmatize(tok) for tok in tokens]
        out.append(" ".join(lemmas))
    return out

all_keywords_lemmatized = lemmatize_list(all_keywords)

print("lemmatized 关键词总数:", len(all_keywords_lemmatized))
print(all_keywords_lemmatized[:20])


#%%

# -------- A. Sentence Split --------
"""
    This function splits a block of text into sentences and returns each sentence 
    along with its character-level start and end positions (spans) in the original text.

    Parameters:
        text (str): The input text string.

    Returns:
        List[Tuple[str, int, int]]: A list of tuples where each tuple contains:
            - The sentence string
            - The start index of the sentence in the original text
            - The end index of the sentence in the original text
"""
def bling_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    # Check if input is a valid non-empty string
    if not isinstance(text, str) or not text:
        return []
    # Use Blingfire to split text into sentences; output is one string separated by '\n'
    s = text_to_sentences(text)
    if not s:
        return []
    # Split into list of sentence strings, removing empty lines
    sentences = [t for t in s.split("\n") if t]

    spans = []
    cursor = 0
    for sent in sentences:
        idx = text.find(sent, cursor)
        if idx == -1:
             # If not found, try again after stripping leading/trailing spaces
            stripped = sent.strip()
            idx = text.find(stripped, cursor)
            if idx == -1:
                # As a last resort, use regex search from the cursor position
                m = re.search(re.escape(stripped), text[cursor:])
                if not m:
                    continue
                idx = cursor + m.start()
            sent = stripped
        start = idx
        end = idx + len(sent)
        spans.append((sent, start, end))
        cursor = end
    return spans
#%%


# ---- 辅助函数：句子 Lemmatization ----
def lemmatize_sentence(sentence: str) -> str:
    """
    对句子进行分词+lemmatization（默认noun pos）
    """
    tokens = re.findall(r"\b\w+\b", str(sentence).lower())  # 简单分词
    lemmas = [lemm.lemmatize(tok) for tok in tokens]       # 默认pos='n'
    return " ".join(lemmas)

# -------- B. Compile keyword regex pattern (with optional word boundaries & case insensitivity) --------
"""
    Create a compiled regex pattern to match any of the given keywords.

    Parameters:
        keywords (Iterable[str]): List or set of keyword strings to match.
        case_insensitive (bool): Whether to ignore case when matching (default: True).
        word_boundary (bool): Whether to require word boundaries around each keyword (default: True).

    Returns:
        re.Pattern: A compiled regex pattern that matches any of the keywords.
                    If no valid keywords are given, returns a pattern that matches nothing.
"""

def compile_keyword_pattern(keywords: Iterable[str], case_insensitive: bool = True, word_boundary: bool = True) -> re.Pattern:
    # Escape and clean all keywords: strip whitespace and escape special regex characters
    kws = [re.escape(str(k).strip()) for k in keywords if k and str(k).strip()]

    # If no valid keywords, return a regex that never matches
    if not kws:
        return re.compile(r"(?!x)x")  # Matches nothing
    # Join keywords using | for "OR" matching
    body = "|".join(kws)
    # Add word boundaries 
    if word_boundary:
        body = rf"\b(?:{body})\b"
    flags = re.IGNORECASE if case_insensitive else 0
    # Compile regex with or without case-insensitive flag
    return re.compile(body, flags)


# -------- C. 合并重叠窗口（句子索引上的区间）--------

def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged
"""
ranges = [(1,3), (2,6), (8,10), (9,11)] 
运行过程： 排序后：[(1,3), (2,6), (8,10), (9,11)] 
初始 merged = [(1,3)] 遍历 (2,6)：因为 2 <= 3 → 合并成 (1,6) merged = [(1,6)] 
遍历 (8,10)：因为 8 > 6 → 不重叠 → 加入 merged = [(1,6), (8,10)] 
遍历 (9,11)：因为 9 <= 10 → 合并成 (8,11) merged = [(1,6), (8,11)]
"""
# -------- D. 单条文本：命中 + 窗口抽取 --------
def extract_windows_for_text(
    text: str,
    pattern: re.Pattern,
    window: int = 1,
    merge_overlaps: bool = True
) -> List[Dict[str, Any]]:
    spans = bling_sentences_with_spans(text)
    if not spans:
        return []

    raw_sents = [s for s, _, _ in spans]                   # 原始句子
    lemm_sents = [lemmatize_sentence(s) for s in raw_sents] # lemma 句子

    # 用 lemma 匹配
    hits = [i for i, sent in enumerate(lemm_sents) if pattern.search(sent)]
    if not hits:
        return []

    raw_ranges = []
    for i in hits:
        l = max(0, i - window)
        r = min(len(raw_sents), i + window + 1)
        raw_ranges.append((l, r))

    ranges = merge_ranges(raw_ranges) if merge_overlaps else raw_ranges

    out = []
    for l, r in ranges:
        mid = (l + r - 1) // 2
        cand = sorted([i for i in hits if l <= i < r], key=lambda x: (abs(x - mid), x))
        center = cand[0] if cand else mid

        out.append({
            "center_idx": center,
            "center_sentence": lemm_sents[center],    # ✅ 仍然是 lemma 版本
            "window_sentences": raw_sents[l:r],       # ✅ 返回原文窗口
            "window_left_idx": l,
            "window_right_idx": r-1,
        })
    return out

#%%

# -------- E. 面向 DataFrame 的批处理（只展开命中窗口）--------
from tqdm import tqdm

def step1_extract_keyword_windows(
    df: pd.DataFrame,
    text_col: str,
    keywords: Iterable[str],
    window: int = 1,
    case_insensitive: bool = True,
    word_boundary: bool = False,   # bigram 建议 False
    merge_overlaps: bool = True,
    return_center_only: bool = False,
) -> pd.DataFrame:
    df2 = df.copy()
    df2["_orig_index"] = df.index
    pat = compile_keyword_pattern(keywords, case_insensitive, word_boundary)

    wins_list = []
    for t in tqdm(df2[text_col].astype(str), total=len(df2), desc="Extracting keyword windows"):
        wins_list.append(extract_windows_for_text(t, pat, window=window, merge_overlaps=merge_overlaps))
    df2["_wins"] = wins_list

    out = df2.explode("_wins")
    out = out[out["_wins"].notna()]

    keys = ["center_idx","center_sentence","center_start","center_end",
            "window_sentences","window_left_idx","window_right_idx","window_start","window_end"]
    tmp = pd.DataFrame(out["_wins"].tolist(), index=out.index)
    tmp = tmp.reindex(columns=keys, fill_value=None)
    out[keys] = tmp
    out = out.drop(columns=["_wins"])

    if return_center_only:
        keep_cols = ["_orig_index","center_sentence","center_start","center_end"]
        keep_cols += [c for c in df.columns if c != text_col]
        out = out[keep_cols]

    out = out.reset_index(drop=True)
    return out





# %%
import os
import math

batch_size = 5000
n_batches = math.ceil(len(call_clean) / batch_size)

for i in range(n_batches):
    save_path = f"C:/data/JF_context_batch_{i+1}.parquet"
    if os.path.exists(save_path):
        print(f"跳过 batch {i+1}, 已存在: {save_path}")
        continue

    start = i * batch_size
    end = min((i+1) * batch_size, len(call_clean))
    batch = call_clean.iloc[start:end]

    print(f"Processing batch {i+1}/{n_batches} ({start}:{end})")

    out = step1_extract_keyword_windows(
        df=batch,
        text_col="componenttext",      
        keywords=all_keywords_lemmatized,       
        window=1,                        
        case_insensitive=True,          
        word_boundary=False,              
        merge_overlaps=True,             
        return_center_only=False         
    )

    out.to_parquet(save_path, index=False)

print("所有未完成的批次都已处理 ✅")
from glob import glob

files = sorted(glob("C:/data/JF_context_batch_*.parquet"))
dfs = [pd.read_parquet(f) for f in files]
JF_context = pd.concat(dfs, ignore_index=True)

print(JF_context.shape)
JF_context.to_parquet("C:/data/JF_context.parquet", index=False)

# %%
batch9 = pd.read_parquet("C:/data/JF_context_batch_9.parquet")

#%%
pd.set_option("display.max_colwidth", None) 
batch9['window_sentences'].head()

#%%
sample = batch9.head(10)

# %%
# --- Code test: sample + keyword window extraction ---
sample1 = call_clean[call_clean["componenttext"].notna()].sample(
    n=10, random_state=44
).reset_index(drop=True)
#%%
wins_center = step1_extract_keyword_windows(
    df=sample1,
    text_col="componenttext",      
    keywords=all_keywords_lemmatized,       
    window=1,                        
    case_insensitive=True,          
    word_boundary=True,              
    merge_overlaps=True,             
    return_center_only=False  
)

print(wins_center.shape)
#%%
pd.set_option("display.max_colwidth", None)  # 不截断列内容
print(wins_center['center_sentence'].head())
#%%
pd.set_option("display.max_colwidth", None)  # 不截断列内容
print(wins_center['window_sentences'].head())

# %%
def find_matched_keywords(text, keywords):
    matched = [kw for kw in keywords if kw in text]
    return matched

wins_center['matched_keywords'] = wins_center['center_sentence'].apply(
    lambda x: find_matched_keywords(str(x), all_keywords_lemmatized)
)
#%%
wins_center[['center_sentence','matched_keywords']]

# %%
lemma2orig = {}
for orig in all_keywords:
    lem = " ".join([lemm.lemmatize(tok) for tok in orig.lower().split()])
    if lem not in lemma2orig:
        lemma2orig[lem] = []
    lemma2orig[lem].append(orig)
# %%
test_word = "ion solution"
print("在原始字典里的候选：", lemma2orig.get(test_word, []))

# %%
[w for w in regulatory if "ion solution" in w.lower()]
# %%
sample = [
    "The newly engineered **ion solution** dramatically increases the charge retention of lithium-ion batteries.",
    "To address the degradation issue, we proposed an advanced ion solution that stabilizes electrolyte reactions.",
    "While most traditional methods rely on solid-state chemistry, our team focused on optimizing the liquid ion solution formula.",
    "This paper evaluates the thermal stability of the ion solution under various storage conditions.",
    "They patented a next-gen ion solution designed specifically for portable medical devices and wearables."
]
df_sample = pd.DataFrame({'componenttext': sample})
#%%
wins_center = step1_extract_keyword_windows(
    df=df_sample,
    text_col="componenttext",      
    keywords=all_keywords_lemmatized,       
    window=1,                        
    case_insensitive=True,          
    word_boundary=True,              
    merge_overlaps=True,             
    return_center_only=False  
)
# %%
pd.set_option("display.max_colwidth", None)  # 不截断列内容
print(wins_center['center_sentence'].head())
# %%
