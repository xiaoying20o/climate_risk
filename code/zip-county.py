#%%
import pandas as pd
import numpy as np
# %%
#------------- Load  Data
call = pd.read_parquet("D:/data/transcript_permno.parquet")

#%%
location = pd.read_csv("D:/data/location.csv")
#%%
location.head()
# %%
#------ Merge call and Headquater Location----------------------
call = call[call['earnings_type'] == 1]
call['date'] = pd.to_datetime(call['date'])
location['datadate'] = pd.to_datetime(location['datadate'])
call = call.sort_values(["date"])
location = location.sort_values(["datadate"])
merged = pd.merge_asof(
    call,
    location,
    by="gvkey",
    left_on="date",
    right_on="datadate",
    direction="backward"   
)
merged = merged[merged['loc'] == 'USA'] # Keep only companies located in the USA
merged.columns
merged.shape

# %%
#--------------Clean Zip code----------------
merged['zip5'] = merged['addzip'].astype(str).str.strip()
merged['zip5'] = merged['zip5'].str[:5]
merged['zip5'] = merged['zip5'] .str.extract(r'(\d{5})')[0]
#fill in 5 digit with 0
merged['zip5'] = merged['zip5'].str.zfill(5)
print(merged[['addzip', 'zip5']].head())
# %%
#------------Contact crosswalk file---------
import glob
import re

# path
path = "D:/data/crosswalk/ZIP_COUNTY_*.xlsx"

files = glob.glob(path)

crosswalk_list = []
for f in files:
    match = re.search(r'(\d{6})', f) #eg:032010
    if match:
        datecode = match.group(1)  # e.g. '032010'
        month = int(datecode[:2])  # 前两位 = 月份
        year = int(datecode[2:])   # 后四位 = 年份
        quarter = (month - 1)//3 + 1
        quarter_str = f"{year}Q{quarter}"

        df_temp = pd.read_excel(f)
        df_temp['quarter'] = quarter_str
        crosswalk_list.append(df_temp)

# contact crosswalk
crosswalk_all = pd.concat(crosswalk_list, ignore_index=True)
crosswalk_all['zip5'] = crosswalk_all['ZIP'].astype(str).str.strip().str.zfill(5)






"""
We first merge county codes using the quarterly crosswalk files for dates after 2010.  
For earnings calls dated before 2010, we use the 2010 Q1 crosswalk as a proxy.

After inspecting the merged results, we find a significant number of missing values.  
This is likely because the crosswalk data often updates ZIP code-to-county mappings 
with a lag of about two years relative to the earnings call date.

To address this issue, we later switch to using `merge_asof` for more accurate matching.
"""






#%%
#------ Because `merge_asof` only keeps the first ZIP–county match (hiding the one-to-many relationship), 
# we pre-aggregate all county-level information for each ZIP code and quarter.
crosswalk_all_grouped = (
    crosswalk_all
    .groupby(['zip5', 'quarter_date'])
    .apply(lambda g: pd.Series({
        'COUNTY': ','.join(g.sort_values('COUNTY')['COUNTY'].astype(str)),
        'RES_RATIO': ','.join(g.sort_values('COUNTY')['RES_RATIO'].astype(str)),
        'BUS_RATIO': ','.join(g.sort_values('COUNTY')['BUS_RATIO'].astype(str)),
        'OTH_RATIO': ','.join(g.sort_values('COUNTY')['OTH_RATIO'].astype(str)),
        'TOT_RATIO': ','.join(g.sort_values('COUNTY')['TOT_RATIO'].astype(str)),
    }))
    .reset_index()
)

crosswalk_all_grouped['quarter_date'] = crosswalk_all_grouped['quarter_date'].dt.date
crosswalk_all_grouped['ZIP'] = crosswalk_all_grouped['ZIP'].astype(str)
crosswalk_all_grouped.to_parquet("D:/data/crosswalk_all_grouped.parquet")



#%%
#----------------- # Use merge_asof to perform a nearest-time match by zip5 and quarter_date----------------
crosswalk_all_grouped = pd.read_parquet("D:/data/crosswalk_all_grouped.parquet")
crosswalk_all_grouped['quarter_date'] = pd.to_datetime(crosswalk_all_grouped['quarter_date'])
merged['quarter_date'] = merged['datadate']
merged = merged.sort_values(by=['quarter_date'])
crosswalk_all_grouped = crosswalk_all_grouped.sort_values(by=['quarter_date'])
map_without_time = pd.merge_asof(
    merged,
    crosswalk_all_grouped,
    by="zip5",
    on="quarter_date",
    direction="nearest"
)
map_without_time.to_parquet("D:/data/map_without_time.parquet")


# %%
#--------------------- Check unmatched code-------------
unmatched_rows = map_without_time[map_without_time['COUNTY'].isna()].copy()
unmatched_rows.shape
unmatched_zip = unmatched_rows.dropna(subset=['gvkey','zip5']).drop_duplicates(subset=['gvkey','zip5'])
unmatched_zip.columns
unmatched_zip = unmatched_zip.drop(columns=['componenttext','altprc','vol','mktcap','costat','curcd','datafmt','indfmt','consol'])
unmatched_zip.to_csv("D:/data/unmatched_zip.csv")

#%%
#------------- After Manully check the county code, Merge unmatched zip back--------
find_match = pd.read_csv("D:/data/unmatched_zip.csv")
find_match['zip5'] = find_match['zip5'].astype(str)
find_match['COUNTY'] = find_match['COUNTY'].astype("str")
map_without_time = map_without_time.merge(
    find_match[['gvkey','zip5','COUNTY']],
    on=['gvkey','zip5'],
    how='left',
    suffixes=('', '_new'),   
    indicator=True
)

map_without_time['COUNTY'] = map_without_time['COUNTY_new'].combine_first(map_without_time['COUNTY'])
map_without_time = map_without_time.drop(columns=['COUNTY_new'])
# check merge
unmatched_rows = map_without_time[map_without_time['COUNTY'].isna()].copy()
unmatched_rows.shape

#%%
map_without_time.to_parquet("D:/data/map_without_time.parquet")

#%%
#---------------- Chcek 1vN map---------------
multi_county_rows = map_without_time[map_without_time['COUNTY'].str.contains(',', na=False)]
multi_county_rows[['gvkey','date','permno','companyname','zip5','COUNTY','quarter_date','RES_RATIO','BUS_RATIO','OTH_RATIO','TOT_RATIO']].head(10)


# %%
#-------- Check Ratio------------
df = multi_county_rows.copy()
df['county_list'] = df['COUNTY'].astype(str).str.split(',')
df['tot_ratio_list'] = df['TOT_RATIO'].astype(str).str.split(',')
# explode
df = df.explode(['county_list', 'tot_ratio_list'])
df['tot_ratio_list'] = df['tot_ratio_list'].astype(float)

# %%
# Step 1: 每个 zip5+quarter+county 的比例求和
dominant = (
    df.groupby(['zip5', 'quarter_date', 'county_list'], as_index=False)['tot_ratio_list']
      .sum()
)

# Step 2: 在每个 zip5+quarter 内找最大 tot_ratio 的 county
dominant = dominant.loc[
    dominant.groupby(['zip5', 'quarter_date'])['tot_ratio_list'].idxmax()
]

# Step 3: 标记是否 >= 0.8
dominant['is_dominant'] = dominant['tot_ratio_list'] >= 0.8
dominant[['zip5','county_list','quarter_date','tot_ratio_list','is_dominant']].head(10)
ratio = dominant['is_dominant'].mean()
print(f"有 dominant county 的比例: {ratio:.2%}")



# %%
#--------------Map County Code via Zip-------------
"""
Here we merge county code by quaterly crosswalk after 2010. 
For thoes which are before 2010, we merged based on 2010 Q1
"""
merged['year'] = merged['datadate'].dt.year.astype('Int64')
merged['quarter'] = merged['datadate'].dt.quarter.astype('Int64')
merged['year_quarter'] = merged['year'].astype(str) + "Q" + merged['quarter'].astype(str)



# split Data into Pre 2010 and Post 2010
df_pre2010  = merged[merged['year'] < 2010].copy()
df_post2010 = merged[merged['year'] >= 2010].copy()
crosswalk_2010Q1 = crosswalk_all[crosswalk_all['quarter']=="2010Q1"]
merged_county_post2010 = df_post2010.merge(
    crosswalk_all,
    left_on=['zip5','year_quarter'],
    right_on=['zip5','quarter'],
    how='left',
    indicator=True
)


merged_county_pre2010 = df_pre2010.merge(
    crosswalk_2010Q1,
    how="left",
    on="zip5",
    indicator=True
)


merged_county_all = pd.concat(
    [merged_county_pre2010, merged_county_post2010],
    ignore_index=True
)
#%%
# ---------Check merge result--------
# Find the rows that failed to merge (i.e., COUNTY not matched)
not_merged_rows = merged_county_all[merged_county_all['_merge'] == 'left_only']
print(f"There are {not_merged_rows.shape[0]} rows that failed to match with the crosswalk.")
not_merged_rows[['zip5', 'date', 'add1', 'addzip', 'city']].head()

# Get unique unmatched ZIP5 codes
unmatched_zips = not_merged_rows['zip5'].dropna().unique()
print(f"There are {len(unmatched_zips)} unique ZIP5 codes not found in the crosswalk(When match time as well).")
# Check which of those ZIP5s actually exist in crosswalk_all
crosswalk_check = crosswalk_all[crosswalk_all['zip5'].isin(unmatched_zips)]
print(f"Among those ZIP5s, {crosswalk_check['zip5'].nunique()} are present in the crosswalk(without time).")

crosswalk_check[crosswalk_check['zip5']=='95052']