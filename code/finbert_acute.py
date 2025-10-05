###### This file should be applied at Python 3.12.11
#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import torch
import ast
import pandas as pd
import numpy as np, textwrap
import pyarrow as pa
import pyarrow.parquet as pq
#%%
# --- Test CUDA availability with PyTorch ---
import torch

print("PyTorch version:", torch.__version__)
print("Is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))
    print("CUDA version used for PyTorch build:", torch.version.cuda)



#%%
# ==== FinBERT 一键推理（center / window 可切换）====


MODEL = "yiyanghkust/finbert-tone"

def _to_list(maybe_list):
    """
    Normalize input into a list of strings.
    This ensures that 'window_sentences' (or similar fields) 
    always become List[str], regardless of the original format.

    Handles multiple cases:
      - numpy.ndarray  → convert to list of str
      - Python list    → convert each element to str
      - String that looks like a list (e.g. "[a, b, c]") → safely eval to list
      - Fallback (single element) → wrap into a list
    """
    
    # numpy.ndarray → list
    if isinstance(maybe_list, np.ndarray):
        return [str(s) for s in maybe_list if s]
    
    # Python list → convert elements to str
    if isinstance(maybe_list, list):
        return [str(s) for s in maybe_list if s]
    
    #  String that looks like a list (e.g. "[...]", from CSV/Parquet storage)
    if isinstance(maybe_list, str) and maybe_list.startswith("["):
        try:
            # Safely evaluate the string into a Python object (list, tuple, etc.)
            return [str(s) for s in ast.literal_eval(maybe_list) if s]
        except Exception:
            # If parsing fails, just return the original string inside a list
            return [maybe_list]
    # Fallback → treat as a single element and wrap in a list
    return [str(maybe_list)]

def load_finbert():
    """自动检测 GPU；CUDA 下用 fp16 加速"""
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else None
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=dtype)
    clf = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=device)
    return clf

def run_finbert(df: pd.DataFrame,
                mode: str = "center",     # "center" or "window"
                batch_size: int = 64,
                max_length: int = 512,
                add_signed: bool = True,
                progress_desc: str = None) -> pd.DataFrame:
    """
    mode="center": only center_sentence  
    mode="window": only window_sentences 
    """
    assert mode in {"center", "window"}
    clf = load_finbert()

    if mode == "center":
        texts = df["center_sentence"].astype(str).tolist()
        desc = progress_desc or "FinBERT (center)"
    else:
        # compile windows sentence together（note that windows contains center sentence）
        texts = [" ".join(_to_list(ws)) for ws in df["window_sentences"].tolist()]
        desc = progress_desc or "FinBERT (window)"

    labels, scores = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch = texts[i:i+batch_size]
        out = clf(batch, batch_size=len(batch), truncation=True, max_length=max_length)
        labels.extend([o["label"] for o in out])
        scores.extend([float(o["score"]) for o in out])

    out_df = df.copy()
    out_df["finbert_label"] = labels
    out_df["finbert_score"] = [round(s, 4) for s in scores]

    if add_signed:
        def signed(lab, s):
            lab = (lab or "").lower()
            if lab.startswith("pos"): return +s
            if lab.startswith("neg"): return -s
            return 0.0
        out_df["finbert_signed"] = [round(signed(l, v), 4) for l, v in zip(labels, scores)]

    return out_df



# %%
# read parquet 
df = pd.read_parquet("D:/data/acute_context.parquet")

# print
print(df.head())


#%%
#test code
df_sample = df.sample(n=100, random_state=42)  
df_window = run_finbert(df_sample, mode="window", batch_size=64)
df_window.head()

# %%
# apply for the whole dataset
finbert =  run_finbert(df, mode="window", batch_size=64)



# %%
finbert.head()

# %%


def join_ws(x):
    if isinstance(x, np.ndarray): x = x.tolist()
    if isinstance(x, list): return " ".join(map(str, x))
    return str(x)

def show_samples(df, n=5, width=120):
    for i, row in df.head(n).iterrows():
        window_text = join_ws(row["window_sentences"])
        print(f"\n── Sample #{i} ─────────────────────────────────────────────────")
        print("CENTER:", row["center_sentence"])
        print("WINDOW:\n", textwrap.fill(window_text, width=width))
        if "finbert_label" in df and "finbert_score" in df:
            print(f"LABEL: {row['finbert_label']} | SCORE: {row['finbert_score']}")
show_samples(finbert, n=5, width=100)

# %%
finbert.to_parquet("D:/data/finbert_acute.parquet", index=False)
# %%
