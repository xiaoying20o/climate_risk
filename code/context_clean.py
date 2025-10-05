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
def safe_clean_text(raw_text):
    if isinstance(raw_text, str):
        return clean_text(raw_text)
    return raw_text


#%%
