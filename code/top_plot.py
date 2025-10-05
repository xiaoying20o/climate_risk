#%%
import pandas as pd



#%%
top = pd.read_csv('D:/projects/climate_risk/top100_bigrams.csv')
# %%
top.shape

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
# %%
def check_belonging_priority(word, small_dicts, total_dict):
    matched = [name for name, d in small_dicts.items() if word in d]
    if matched:
        return matched
    elif word in total_dict:
        return ["total"]
    else:
        return []

# small
small_dicts = {
    "physical": set(physical),
    "opportunity": set(opportunity),
    "regulatory": set(regulatory)
}

# total 
total_dict = set(total)
top['in_dicts'] = top['bigram'].apply(lambda x: check_belonging_priority(x, small_dicts, total_dict))
top.head()
# %%
import matplotlib.pyplot as plt

#
top['dict_label'] = top['in_dicts'].apply(lambda x: x[0] if x else 'none')

# 
label_counts = top['dict_label'].value_counts()
plt.figure(figsize=(8, 6))
label_counts.plot.bar(color='skyblue', edgecolor='black')

plt.title('Number of Bigrams by Dictionary Category')
plt.xlabel('Dictionary Category')
plt.ylabel('Number of Bigrams')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#%%
import plotly.express as px

top_100 = top[['bigram', 'in_dicts','count']].head(100).copy()
top_100['dict_label'] = top_100['in_dicts'].apply(lambda x: x[0] if x else 'none')
top_100['bigram'] = pd.Categorical(top_100['bigram'], categories=top_100['bigram'][::-1], ordered=True)


fig = px.scatter(
    top_100,
    y='bigram',
    x='dict_label',
    color='dict_label',
    title='Top 100 Bigrams and Their Dictionary Category',
    labels={'dict_label': 'Dictionary Category', 'bigram': 'Bigram'},
    height=2000  
)

fig.update_traces(marker=dict(size=12))


fig.update_layout(
    yaxis=dict(
        categoryorder='array',
        categoryarray=top_100['bigram'].tolist()
    )
)

fig.show()

# %%

# %%
def sum_count_for_regulatory(df):
    return df[df['dict_label'] == 'regulatory']['count'].sum()
total_regu = sum_count_for_regulatory(top_100)
print(total_regu)
# %%
top.to_csv("top_bigram.csv", index=False)
# %%
