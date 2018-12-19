import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
import re
from textwrap import wrap

# from geopandas import GeoSeries, GeoDataFrame
#website for colors https://www.w3schools.com/colors/colors_picker.asp
#pip install pysal

def make_heat(df):
    '''Turn numeric codes into characters to be able to merge with the world dataframe'''
    s = "4: Afghanistan 5: Albania 6: Algeria 7: Andorra 8: Angola 10: Antigua and Barbuda 11: Argentina 12: Armenia 14: Australia 15: Austria 16: Azerbaijan 17: Bahamas 18: Bahrain 19: Bangladesh 20: Barbados 21: Belgium 22: Belize 23: Benin 24: Bermuda 25: Bhutan 26: Bolivia 28: Bosnia-Herzegovina 29: Botswana 30: Brazil 31: Brunei 32: Bulgaria 33: Burkina Faso 34: Burundi 35: Belarus 36: Cambodia 37: Cameroon 38: Canada 40: Cayman Islands 41: Central African Republic 42: Chad 43: Chile 44: China 45: Colombia 46: Comoros 47: Republic of the Congo 49: Costa Rica 50: Croatia 51: Cuba 53: Cyprus 54: Czech Republic 55: Denmark 56: Djibouti 57: Dominica 58: Dominican Republic 59: Ecuador 60: Egypt 61: El Salvador 62: Equatorial Guinea 63: Eritrea 64: Estonia 65: Ethiopia 66: Falkland Islands 67: Fiji 68: Finland 69: France 70: French Guiana 71: French Polynesia 72: Gabon 73: Gambia 74: Georgia 75: Germany 76: Ghana 77: Gibraltar 78: Greece 79: Greenland 80: Grenada 81: Guadeloupe 83: Guatemala 84: Guinea 85: Guinea-Bissau 86: Guyana 87: Haiti 88: Honduras 89: Hong Kong 90: Hungary 91: Iceland 92: India 93: Indonesia 94: Iran 95: Iraq 96: Ireland 97: Israel 98: Italy 99: Ivory Coast 100: Jamaica 101: Japan 102: Jordan 103: Kazakhstan 104: Kenya 106: Kuwait 107: Kyrgyzstan 108: Laos 109: Latvia 110: Lebanon 111: Lesotho 112: Liberia 113: Libya 114: Liechtenstein 115: Lithuania 116: Luxembourg 117: Macau 118: Macedonia 119: Madagascar 120: Malawi 121: Malaysia 122: Maldives 123: Mali 124: Malta 125: Man, Isle of 127: Martinique 128: Mauritania 129: Mauritius 130: Mexico 132: Moldova 134: Mongolia 136: Morocco 137: Mozambique 138: Myanmar 139: Namibia 141: Nepal 142: Netherlands 143: New Caledonia 144: New Zealand 145: Nicaragua 146: Niger 147: Nigeria 149: North Korea 151: Norway 152: Oman 153: Pakistan 155: West Bank and Gaza Strip 156: Panama 157: Papua New Guinea 158: Paraguay 159: Peru 160: Philippines 161: Poland 162: Portugal 163: Puerto Rico 164: Qatar 166: Romania 167: Russia 168: Rwanda 173: Saudi Arabia 174: Senegal 175: Serbia-Montenegro 176: Seychelles 177: Sierra Leone 178: Singapore 179: Slovak Republic 180: Slovenia 181: Solomon Islands 182: Somalia 183: South Africa 184: South Korea 185: Spain 186: Sri Lanka 189: St. Kitts and Nevis 190: St. Lucia 192: St. Martin 195: Sudan 196: Suriname 197: Swaziland 198: Sweden 199: Switzerland 200: Syria 201: Taiwan 202: Tajikistan 203: Tanzania 204: Togo 205: Thailand 206: Tonga 207: Trinidad and Tobago 208: Tunisia 209: Turkey 210: Turkmenistan 213: Uganda 214: Ukraine 215: United Arab Emirates 216: Great Britain 217: United States 218: Uruguay 219: Uzbekistan 220: Vanuatu 221: Vatican City 222: Venezuela 223: Vietnam 226: Wallis and Futuna 228: Yemen 229: Democratic Republic of the Congo 230: Zambia 231: Zimbabwe 233: Northern Ireland 235: Yugoslavia 236: Czechoslovakia 238: Corsica 334: Asian 347: East Timor 349: Western Sahara 351: Commonwealth of Independent States 359: Soviet Union 362: West Germany (FRG) 377: North Yemen 403: Rhodesia 406: South Yemen 422: International 428: South Vietnam 499: East Germany (GDR) 520: Sinhalese 532: New Hebrides 603: United Kingdom 604: Zaire 605: People's Republic of the Congo 999: Multinational 1001: Serbia 1002: Montenegro 1003: Kosovo 1004: South Sudan"

    country_list = re.split('(\d+)',s)
    length = len(country_list)
    country_list_len = range(length)
    numbers = country_list[1::2]
    countries = country_list[2::2]

    new_countries = []
    for country in countries:
        # print(type(country))
        new = country[2::]
        new2 = new[:-1]
        new_countries.append(new2)

    new_numbers = []
    for number in numbers:
        new = int(number)
        # print(new)
        new_numbers.append(new)

    dict_country = dict(zip(new_numbers, new_countries))
    df['country_names'] = df['country'].replace(dict_country)
    df = df[['country','country_names']]
    df.columns=['id', 'name']
    count_countries(df)

def count_countries(df):
    '''
    Counting the number of country instances, across years
    df: original pandas df
    '''
    df_base_count = df.groupby('name').count()
    df_base_count = df_base_count.reset_index()
    df_base_count.rename(columns ={'id':'count'}, inplace = True)
    df_base_count = df_base_count.sort_values('count', ascending = False)
    df_base_count.reset_index(drop=True, inplace = True)
    merge_plot_heat(df_base_count)

def merge_plot_heat(df):
    '''
    df: df from count_countries
    '''
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    country_names = world[['name', 'continent', 'geometry', 'iso_a3']]
    world_merged = country_names.merge(df, how='left', on='name')
    world_merged.fillna(value=0,inplace=True)
    world_merged.plot(column='count', cmap='Oranges', linewidth=0.5, edgecolor='black', legend=True,
     figsize=(20,20), scheme='fisher_jenks')
    plt.title('Suicide Bombing Frequency in 2017', weight='bold', fontsize=25)
    plt.savefig('images/Heatmap')

def sep_dfs(df, group_list):
    '''
    This function will take in an original dataframe, and create separate dataframes for all groups in group_list
    df: original dataframe
    group_list: list of groups to observe
    base_str: country of interest, defaults to United States
    '''
    df_list = []
    for str in group_list:
        df_for_merge = df[(df['suicide_text'].str.contains(str))]
        df_list.append(df_for_merge)
    return df_list

def sum_one_var(df_list, groupby, to_sum):
    '''
    This function will take in a list of dataframes, each for a different group, group by some variable (groupby),
    and sum up another variable (to_sum), and return the grouped and summed dfs
    df_list: list of dfs created in sep_dfs
    groupby: variable to group on
    to_sum: variable to sum up
    '''
    sum_list = []
    for df in df_list:
        grouped = df.groupby(groupby).count()[to_sum]
        sum_list.append(grouped)
    return sum_list

def multi_line_plot(df_sum_list, legend_list, title, xlab, ylab, save_bool, plot_bool, save_as, legend_title, xrestrict_lower=0, xrestrict_upper=0, yrestrict_upper=0):
    '''
    This function will create a set of line graphs in a single space.
    df_sum_list: list of dfs that have been grouped and summed on some variable in function sum_var
    group_list: list of groups/categories to observe (this is the multi-category variable in the legend)
    title: title for plot
    xlab: x axis label for plot
    ylab: y axis label for plot
    save_bool: 1 == save the plot, 0 == don't save it
    plot_bool: 1 == display the plot, 0 == don't display it
    '''
    colors = ['#cc6600', '#336699']
    alphas = [.6, .2]
    for df, item, color, alpha in zip(df_sum_list, legend_list, colors, alphas):
        line = df.plot(kind = 'area', label=item, alpha=alpha, linewidth=2.0, color=color)
        line.set_label(item)
    plt.legend(title=legend_title, loc='upper left')
    plt.title(title, weight='bold')
    if xrestrict_upper > 0:
        plt.xlim(xrestrict_lower, xrestrict_upper)
    if yrestrict_upper > 0:
        plt.ylim(0, yrestrict_upper)
    plt.xlabel(xlab, weight='bold')
    plt.ylabel(ylab, weight='bold')
    if save_bool == 1:
        plt.savefig(save_as)
    if plot_bool == 1:
        plt.show()

def make_pie(df):
    '''Make pie chart to display proportion of groups that engaged in suicide bombings in 2017'''
    df_suicide = df[df.suicide==1]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    labels = ['No Suicide Bombings', 'Some Suicide Bombings']
    unique_groups = df.gname.unique().shape[0]
    yes_suicide = df_suicide.gname.unique().shape[0]
    no_suicide = unique_groups - yes_suicide

    sizes = [no_suicide, yes_suicide]
    explode = (0, 0.2)  # only "explode" the 2nd slice
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            startangle=330)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Proportion of Groups that Performed Suicide Bombings in 2017', weight='bold')
    plt.tight_layout()
    # plt.show()
    plt.savefig('images/pie')

def make_stacked_bar(df, zoom=0):
    '''Stacked Bar Chart'''
    '''Remove unknown groups'''
    df = df.drop(df[df.gname == 'Unknown'].index)

    '''Groupby gname and get total attack count, and total suicide count'''
    grouped_total_attacks = df.groupby('gname').size().reset_index()
    print(grouped_total_attacks)
    'the name of the column with values in total_list is named 0'
    total_list = grouped_total_attacks[0].values
    grouped_suicide_attacks = df.groupby('gname').sum()['suicide'].reset_index()
    suicide_list = grouped_suicide_attacks['suicide']

    '''Only plot groups that engaged in suicide bombings in 2017'''
    total = []
    suicide = []
    for i,j in zip(total_list,suicide_list):
        if j > 0:
            total.append(i)
            suicide.append(j)

    nonsuicide = [(i-j) for i,j in zip(total, suicide)]
    zipped = sorted(list(zip(nonsuicide, suicide)), key=lambda x:x[0])
    suicide = [j for i,j in zipped]
    nonsuicide = [i for i,j in zipped]

    '''Plot the proportions'''
    plt.figure(figsize=(12,6))
    N = len(total)
    ind = np.arange(N)    # the x locations for the groups
    p1 = plt.bar(ind, nonsuicide)
    p2 = plt.bar(ind, suicide, bottom=nonsuicide)
    plt.ylabel('Number of Attacks', weight='bold')
    plt.xlabel('Unique Terrorist Groups', weight='bold')
    plt.title('Proportion of Attacks that were Suicide Bombings by Group in 2017', weight='bold', fontsize=20)
    plt.legend((p1[0], p2[0]), ('Non-Suicide Bombing Attacks', 'Suicide Bombings'), loc=2)
    plt.xticks([])
    plt.tight_layout()
    # plt.show()
    if zoom == 1:
        plt.ylim(0,50)
        plt.savefig('images/bar_zoom')
    else:
        plt.savefig('images/bar')

def make_violin(df):
    '''Manipulate order of categories on xaxis'''
    df['suicideR'] = np.where((df['suicide'] == 1), 1, 0)
    '''create object version of suicide variable'''
    dict_suicide = {1: 'Suicide', 0: 'Not suicide'}
    df['suicide_text'] = df['suicideR'].replace(dict_suicide)
    df = df[['suicide_text','nkill','nwound']]
    df.dropna(inplace = True)

    numeric_cols = [col for col in df if df[col].dtype.kind != 'O']
    df[numeric_cols] += 1

    df['log_wound'] = np.log10(df['nwound'])
    df['log_kill'] = np.log10(df['nkill'])

    df_plot = df[['suicide_text','log_kill', 'log_wound']]
    ax = sns.violinplot(x = 'suicide_text', y = 'log_kill', data = df_plot)

    plt.xlabel('Type of Attack', weight='bold')
    plt.ylabel('Log Base 10 Number Killed', weight='bold')
    plt.title('Number of People Killed by Type of Attack in 2017', weight='bold')
    # plt.show()
    plt.savefig('images/killed')
