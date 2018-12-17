import pandas as pd
import code_for_graphing as graph

if __name__ == '__main__':
    '''Original dataframe; toggle this on to create 2017 dataframe'''
    df = pd.read_csv('/Users/janestout/Dropbox/Galvanize/DSI/Capstones/Capstone2_working/GTD/data/globalterrorismdb_0718dist.csv', low_memory=False)


    '''PLOT MULTILINE GRAPH (FIG 1)'''

    '''Create dataframe with only suicide incidence and year'''
    df_lines = df[['suicide', 'iyear']]

    '''Create suicide variable that is text (for legend)'''
    dict_suicide = {1: 'Suicide', 0: 'Not suicide'}
    df_lines['suicide_text'] = df_lines['suicide'].replace(dict_suicide)

    '''Prepare separate dfs for plotting'''
    suicide_text_labels = ['Suicide', 'Not suicide']
    make_separate_dfs = graph.sep_dfs(df_lines, suicide_text_labels)
    make_sums_one_var = graph.sum_one_var(make_separate_dfs, 'iyear', 'suicide_text')

    '''Plot the dfs'''
    graph.multi_line_plot(make_sums_one_var, suicide_text_labels, title='Count of Suicide Bombings over time', xlab='Year', ylab='Event count', save_bool=1, plot_bool=0, save_as='images/suicide_over_time', legend_title='Type of Attack', xrestrict_lower=0, xrestrict_upper=0, yrestrict_upper=17000)



    '''Use 2017 dataframe for all plots that follow'''

    df_2017 = df[df.iyear==2017]
    df_2017.to_csv('data/df_2017.csv', index=False)
    df2017 = pd.read_csv('data/df_2017.csv', low_memory=False)

    '''MAKE PIE GRAPH: PROPORTION OF GROUPS THAT ENGAGED IN SUICIDE BOMBINGS IN 2017 (FIG 2a)'''
    graph.make_pie(df2017)

    '''MAKE STACKED BAR GRAPH: PROPORTION OF ATTACKS THAT ARE SUICIDE BOMBINGS, AMONG GROUPS THAT ENGAGED IN SUICIDE BOMBINGS IN 2017 (FIGS 2b and 2c)'''
    graph.make_stacked_bar(df2017, zoom=1)
    graph.make_stacked_bar(df2017, zoom=0)

    '''MAKE HEATMAP OF 2017 DATA (FIG 3)'''
    graph.make_heat(df2017)

    '''MAKE VIOLIN PLOT OF NKILLED FOR SUICIDE VS NOT SUICIDE IN 2017 (FIG 4)'''
    graph.make_violin(df2017)
