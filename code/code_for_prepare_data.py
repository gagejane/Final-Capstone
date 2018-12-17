import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
import re

def clean_gtd():
    '''This code creates a dataframe containing (a) keys (gname, iyear) in prep for merging with TIOS2 and (b)features for analysis'''
    '''Features were selected based on feature importances > .04 in Capstone 2 Random Forest model'''

    df = pd.read_csv('/Users/janestout/Dropbox/Galvanize/DSI/Capstones/Capstone2_working/GTD/data/globalterrorismdb_0718dist.csv', low_memory=False)

    '''Create binary variables where weapon type == 1 and all else == 0'''
    df['explo_vehicle'] = np.where((df['weapsubtype1'] == 15), 1, 0)
    df['explo_unknown'] = np.where((df['weapsubtype1'] == 16), 1, 0)
    df['firearm_unknown'] = np.where((df['weapsubtype1'] == 5), 1, 0)
    df['explo_project'] = np.where((df['weapsubtype1'] == 11), 1, 0)
    df['explo_other'] = np.where((df['weapsubtype1'] == 17), 1, 0)

    '''Claimed responsibility: code -9 values as missing so that there are only 1 and 0 in this variable'''
    df['claimed'].replace(to_replace=[-9],value=np.NaN, inplace=True)

    '''Hostage taking/Kidnapping: code -9 values as missing so that there are only 1 and 0 in this variable'''
    df['ishostkid'].replace(to_replace=[-9],value=np.NaN, inplace=True)

    '''Create binary variables where specific country == 1 and all else == 0'''
    df['Iraq'] = np.where((df['country'] == 95), 1, 0)
    df['Afghanistan'] = np.where((df['country'] == 4), 1, 0)
    df['India'] = np.where((df['country'] == 92), 1, 0)

    '''Create dataframe for merging with TIOS2'''
    df_suicide_DT=df[['gname', 'iyear', 'claimed', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'explo_project', 'explo_other', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'suicide']]
    df_suicide_DT.dropna(inplace = True)
    df_suicide_DT.to_csv('data/df_GTD_for_merging.csv', index=False)

def merge_dfs():
    '''merge prepared GTD dataframe and TIOS2 on keys: gtd_gname and year'''
    df_gtd = pd.read_csv('data/df_GTD_for_merging.csv', low_memory=False)
    df_TIOS2_full = pd.read_csv("/Users/janestout/Dropbox/Galvanize/DSI/Capstones/FinalWork/Capstone1/data/TIOS_data_v2.csv")
    df_TIOS2_full = df_TIOS2_full[['gtd_gname', 'year', 'religion', 'infrastructure', 'health', 'education', 'finance', 'security', 'social']]
    # df_TIOS2_full.dropna(inplace = True)
    df_TIOS2_full.to_csv('data/df_TIOS2_for_merging.csv', index=False)
    df_TIOS2 = pd.read_csv('data/df_TIOS2_for_merging.csv', low_memory=False)

    '''restructure df_gtd that is the same structure as the TIOS: each row is a 'group-year'''
    grouped = df_gtd.groupby(['gname', 'iyear']).sum().reset_index()
    '''rename gname and iyear so they have the same name as TIOS2'''
    grouped.rename(columns={'gname':'gtd_gname', 'iyear':'year'}, inplace=True)

    df_merged = pd.merge(grouped, df_TIOS2, on=['gtd_gname', 'year'], how='inner')
    df_merged.to_csv('data/df_merged_imbalanced.csv', index=False)
    # print(df_merged['suicide'].value_counts())
#
def upsample():
    '''create a dataframe that upsamples the number of group years with suicide bombing incidents
    to be equal to the number group years that do not have suicide bombing incidents'''
    df = pd.read_csv('data//df_merged_imbalanced.csv', low_memory=False)
    df_majority = df[df.suicide==0]
    df_minority = df[df.suicide>0]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=738, random_state=123)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    df=df_upsampled.copy()

    df=df[['gtd_gname', 'year', 'claimed', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'explo_project', 'explo_other', 'ishostkid', 'Iraq', 'Afghanistan', 'India','religion', 'infrastructure', 'health', 'education', 'finance', 'security', 'social', 'suicide']]

    df.to_csv('data/df_merged.csv', encoding='utf-8', index=False)

def make_count_df(vars):
    '''Keep dataframe in original format where features
    are in count format, due to aggregating into group-years
    This function just renames column names to include _count in the name,
    and creates a binary suicide (target) variable'''
    df = pd.read_csv('data/df_merged.csv', low_memory=False)
    for var in vars:
        new_name = var + '_count'
        df.rename(columns={var:new_name}, inplace=True)
    df['suicide'] = np.where((df['suicide_count'] > 0), 1, 0)
    df.drop(['suicide_count'], axis=1, inplace=True)
    df.to_csv('data/df_merged_count.csv', encoding='utf-8', index=False)
    return df

def make_binary_df(vars):
    '''Make new dataframe that makes features and target binary'''
    df = pd.read_csv('data/df_merged.csv', low_memory=False)
    for var in vars:
        df[var] = np.where((df[var] >= 1), 1, 0)
    df.to_csv('data/df_merged_binary.csv', encoding='utf-8', index=False)
    return df
