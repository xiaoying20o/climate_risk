#%%
import pandas as pd
import numpy as np
# %%
loss_with_call = pd.read_parquet(("D:/data/map_loss"))
county_loss = pd.read_csv("D:/data/County level loss/direct_loss_aggregated_output_25190.csv")
quarterly_finbert = pd.read_parquet("D:/projects/climate_risk/data/quarterly_panel_include_control_next10k.parquet")
#%%
call = pd.read_parquet("D:/data/transcript_permno.parquet")
call['gvkey'] = call['gvkey'].astype(str)
#%%
#-------- Preview Firm Map County Code
map_without_time = pd.read_parquet("D:/data/map_without_time.parquet")
map_without_time = map_without_time.drop(columns=['_merge'])
#%%
firm_county_preview = map_without_time.drop(columns='componenttext').head(100).to_html("preview.html")

#%%
#------------ Preview County-Loss Map gvkey---------
loss_with_call['gvkey'] = loss_with_call['gvkey'].replace('nan', np.nan)
len(loss_with_call['gvkey'])
len(loss_with_call['gvkey'].dropna())
loss_call_preview = (
    loss_with_call[
        (loss_with_call["loss_date"].dt.year > 2005) 
        & (loss_with_call["gvkey"].notna())
    ]
    .drop(columns=[
        'transcriptid', 'headline', 'earnings_type',
        'componenttext', 'altprc', 'vol', 'mktcap', 'costat', 'curcd',
        'datafmt', 'indfmt', 'consol', 'datadate', 'conm','Fatalities', 'FatalitiesPerCapita', 'Fatalities_Duration', 'Injuries'
    ])
    .head(100)
    .to_html("loss_call_preview.html", index=False)
)
#%%
#------ Check Selection Bias-------------

threshold = 2e9 # 1 billion
loss_with_call['exposed_1b'] =np.where(
    (loss_with_call['CropDmg(ADJ 2023)'] + loss_with_call['PropertyDmg(ADJ 2023)'])>threshold,
    1,
    0
)
exposed = loss_with_call[
    (loss_with_call['exposed_1b'] == 1) & 
    (loss_with_call['gvkey'].notna())
]#county确实有上市公司（在我们的call的dataset中） & 有treshold billion以上的loss
check_selection_bias = exposed.merge(
    quarterly_finbert[['gvkey','call_date','finbert_signed_mean']], #这一万个finbert是call中有acute的信息，用来衡量是否有disclosure
    on=['gvkey','call_date'], how='left'
)
check_selection_bias['disclosed'] = check_selection_bias['finbert_signed_mean'].notna().astype(int)
# %%


#%%
check_selection_bias.shape
#%%
(check_selection_bias['disclosed'] == 0).sum()
#%%
check_selection_bias[['COUNTY','CropDmg(ADJ 2023)','PropertyDmg(ADJ 2023)',
                      'loss_date','gvkey','permno','call_date','exposed_1b','disclosed','COUNTY_LIST','TOT_RATIO']]

#%%
check_selection_bias_preview = check_selection_bias[['COUNTY','CropDmg(ADJ 2023)','PropertyDmg(ADJ 2023)',
                      'loss_date','gvkey','permno','call_date','exposed_1b','disclosed','COUNTY_LIST','TOT_RATIO']].head(100).to_html("check_selection_bias_preview.html")


#%%





