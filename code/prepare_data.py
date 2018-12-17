import code_for_prepare_data as prep
import pandas as pd

if __name__ == '__main__':

    '''Get the GTD data ready for merging; create df that contains variables for modeling'''
    prep.clean_gtd()

    '''Merge the cleaned GTD and TIOS2 dfs'''
    prep.merge_dfs()

    '''Deal with class imbalance; suicide bombings are rare, so make their occurance equal to no-suicide bombing occurence'''
    prep.upsample()

    '''Create two dfs for modeling: one that treats features as continuous and one that treats features as binary'''
    to_transform = ['claimed', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'explo_project', 'explo_other', 'ishostkid', 'Iraq', 'Afghanistan', 'India','religion', 'infrastructure', 'health', 'education', 'finance', 'security', 'social', 'suicide']
    prep.make_count_df(to_transform)
    prep.make_binary_df(to_transform)
