#%%
import pandas as pd
import numpy as np

#%%
#-------------- Load Data------------
map_without_time = pd.read_parquet("D:/data/map_without_time.parquet")
county_loss = pd.read_csv("D:/data/County level loss/direct_loss_aggregated_output_25190.csv")


# %%
county_loss['loss_date'] = pd.to_datetime(
    county_loss['Year'].astype(str) + '-' + county_loss['Month'].astype(str) + '-01'
)

county_loss['County_FIPS']=county_loss['County_FIPS'].astype(str)


#%%
county_loss = county_loss.rename(columns={'County_FIPS' : 'COUNTY',
                                          'Year' : 'Loss_Year',
                                          'Month' : 'Loss_Month'})
#%%
map_without_time = map_without_time.drop(columns= '_merge')
map_without_time = map_without_time.rename(columns={'COUNTY' : 'COUNTY_LIST'})
map_without_time = map_without_time.assign(
    COUNTY = map_without_time['COUNTY_LIST'].str.split(',')
).explode('COUNTY')

#%%
# ---------- First identify the firms located within each county by year -----------

"""
Ensure that each county-level loss record (from `county_loss`) is matched with 
the list of firms (gvkeys) that were present in that county during the same year.
This step prepares a firm-year-county exposure dataset for later loss matching.
"""
county_loss = county_loss.rename(columns = {'Loss_Year' : 'year'})
firm_year_exposure = (
    map_without_time.assign(year = map_without_time['date'].dt.year)
    [['gvkey','COUNTY','year']]
    .drop_duplicates()
)
loss_with_firms = county_loss.merge(
    firm_year_exposure,
    on=['COUNTY','year'],
    how='left'
)


#%%
# Check merge 的结果， NA是真的在当年的county中没有相关的gvkey
loss_with_firms['gvkey'].isna().sum()
loss_with_firms[['COUNTY','loss_date','gvkey']].dropna().head(30)
map_without_time[map_without_time['COUNTY'] == '16001']
county_loss[county_loss['COUNTY'] == '16001']

#%%
loss_with_firms['gvkey'] = loss_with_firms['gvkey'].astype(str).str.split('.').str[0] 
map_without_time['gvkey'] = map_without_time['gvkey'].astype(str)
loss_with_firms = loss_with_firms.sort_values('loss_date')
map_without_time = map_without_time.sort_values('date')

loss_with_call = pd.merge_asof(
    loss_with_firms,
    map_without_time,
    left_on='loss_date',
    right_on='date',
    by=['gvkey','COUNTY'],   # 
    direction='forward'
)
#%%
loss_with_call.columns
#%%
loss_with_call=loss_with_call.rename(columns={'date' : 'call_date',
                                              'year' : 'loss_year',
                                              'month' : 'call_month'}
                                              )


loss_with_call.to_parquet("D:/data/map_loss")
#%%
"""
这组数据代表的情况：2023 loss发生的那一年里，gvkey = 156614的公司在这个county在2023 有earnnig call，
但call的数据最后一期直到三月份，所以2023 年5的call merge不到date
call['gvkey'] = call['gvkey'].astype(str)
call[call['gvkey']=='156614']['date']
已经从原始的数据中认证过了,所以这种情况也不能说公司是没有提到
"""
loss_with_call[
    (loss_with_call['COUNTY'] == '48339') & (loss_with_call['gvkey'] == '156614')
][['COUNTY','loss_date','gvkey','date']]

map_without_time[
    (map_without_time['COUNTY'] == '48339') & (map_without_time['gvkey'] == '156614')
][['COUNTY','gvkey','date']]



#%%
# 找出一个县对应多个不同 gvkey 的情况进行merge检验是否每一个county里的公司都对应上了
dup_county = (
    loss_with_firms
    .groupby('COUNTY')['gvkey']
    .nunique()
    .reset_index(name='unique_gvkeys')
)
dup_county = dup_county[dup_county['unique_gvkeys'] > 1]
dup_county.head(10)
map_without_time[map_without_time['COUNTY'] == '1045'][['gvkey','date']].head(20)
loss_with_call[
    (loss_with_call['COUNTY'] == '1045') & (loss_with_call['gvkey'] == '33424')
][['COUNTY','loss_date','gvkey','date']]

#%%
# ------ 这组数据是 Check 是否 一个zip code(firm) 横跨多个county时，每一个county 发生loss 都对应上了最近的那场call
multi_county_rows = map_without_time[map_without_time['COUNTY_LIST'].str.contains(',', na=False)]
multi_county_rows[['gvkey','date','permno','companyname','zip5','COUNTY','quarter_date','RES_RATIO','BUS_RATIO','OTH_RATIO','TOT_RATIO']].head(10)
loss_with_call[
    (loss_with_call['COUNTY'] == '39049') & (loss_with_call['gvkey'] == '63643')
][['COUNTY','loss_date','gvkey','date']]



# %%
