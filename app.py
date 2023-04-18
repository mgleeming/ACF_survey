import os, sys
import pandas as pd
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

st.set_page_config(page_title="ACF survey",layout='wide')

print('==========================================================')
WD = os.getcwd()
DATA_DIR = os.path.join(WD, 'data')

KEY_FILE = os.path.join(DATA_DIR, 'key_2.csv')
DATA_FILE = os.path.join(DATA_DIR, 'data.csv')

# for height of stacked bar charts
FACTOR = 0.25

key_df = pd.read_csv(KEY_FILE, sep ='\t')
data_df = pd.read_csv(DATA_FILE)

# aggregate data
def sum_columns(df, key, new_col_name):
    cols = key_df[key_df['question'].str.contains(key)]['redcap']
    df[new_col_name] = df[cols].sum(axis=1)
    return df

data_df = sum_columns(data_df, 'Your total female', 'total_female_fte') 
data_df = sum_columns(data_df, 'Your total male', 'total_male_fte') 
data_df = sum_columns(data_df, 'academic FTE is', 'total_academic_fte')
data_df = sum_columns(data_df, 'professional FTE is', 'total_professional_fte')
data_df = sum_columns(data_df, 'administrative FTE is', 'total_administrative_fte')
data_df = sum_columns(data_df, 'are on continuing appointments', 'total_continuing_appointments')
data_df = sum_columns(data_df, 'are on fixed-term appointments', 'total_fixed_term_appointments')
data_df = sum_columns(data_df, 'are on casual appointments', 'total_casual_appointments')

# divide total_female_fte by staff_total_fte_cal
data_df['fraction_female'] = data_df['total_female_fte'] / data_df['staff_total_fte_cal'] * 100
data_df['fraction_male'] = data_df['total_male_fte'] / data_df['staff_total_fte_cal'] * 100

data_df['fraction_academic'] = data_df['total_academic_fte'] / data_df['staff_total_fte_cal'] * 100
data_df['fraction_professional'] = data_df['total_professional_fte'] / data_df['staff_total_fte_cal'] * 100
data_df['fraction_administrative'] = data_df['total_administrative_fte'] / data_df['staff_total_fte_cal'] * 100

data_df['fraction_continuing'] = data_df['total_continuing_appointments'] / data_df['staff_total_fte_cal'] * 100
data_df['fraction_fixed_term'] = data_df['total_fixed_term_appointments'] / data_df['staff_total_fte_cal'] * 100
data_df['fraction_casual'] = data_df['total_casual_appointments'] / data_df['staff_total_fte_cal'] * 100

# convert staff_cost to aud if staff_cost_currency == 'NZ Dollars'
data_df['staff_cost_aud'] = data_df['staff_cost']
data_df.loc[data_df['staff_cost_currency'] == 'NZ Dollars', 'staff_cost_aud'] = data_df['staff_cost'] * 0.93

# get rows from key_df that contain 'ms_type'
ms_type_cols = key_df[key_df['key']=='ms_type']['redcap'].unique()

# sum ms_type columns in data_df
data_df['total_instruments'] = data_df[ms_type_cols].sum(axis=1)

instruments = [
    'no_ms_q_ot_it',
    'no_ms_q_ot',
    'no_ms_qtof',
    'no_ms_qqq',
    'no_ms_tof_tof',
    'no_ms_ft_icr',
    'no_ms_it',
    'no_ms_s_q',
    'no_ms_other',
]

for instrument in instruments:
    data_df['total_instrument_count' + instrument] = data_df[instrument] + data_df[instrument + '_ionmob']
    data_df['total_' + instrument] = data_df[instrument] + data_df[instrument + '_ionmob']

data_df['staff_fte_per_instrument'] = data_df['staff_total_fte_cal'] / data_df['total_instruments']

software_names = [ 'HDI', 'SCilS', 'Byonic', 'DIA-NN', 'EncyclopaDIA', 'Mascot', 'Mass Dynamics', 'Mass Hunter', 'Max Quant', 'Metamorpheus', 'MSFragger', 'OpenMS', 'Peaks', 'PeakView/SWATH', 'Progenesis QI for Proteomics ', 'Proteome Discoverer', 'ProteinPilot', 'Scaffold', 'Spectronaut', 'SwathXtend', 'Trans-Proteomics Pipeline', 'Analyst', 'Compound Discoverer', 'ChemStation', 'LipidSearch', 'LipidView', 'Protenesis QI', 'MassLynx ', 'MS-DIAL ', 'MetabolomeExpress', 'FreeStyle', 'GCImage', 'Insight', 'Multiquant', 'Qualbrowser', 'Skyline', 'SciexOS', 'TargetLynx', 'Tracefinder']
software_keys = [ 'soft_hdi', 'soft_scils', 'soft_byonic', 'soft_dia_nn', 'soft_ency', 'soft_mascot', 'soft_md', 'soft_mh', 'soft_mq', 'soft_metamorph', 'soft_msfragger', 'soft_openms', 'soft_peaks', 'soft_swath', 'soft_pqip', 'soft_pd', 'soft_pp', 'soft_scaf', 'soft_spec', 'soft_swathx', 'soft_tpp', 'soft_analyst', 'soft_cd', 'soft_chemstation', 'soft_ls', 'soft_lv', 'soft_pqi', 'soft_mlynx', 'soft_msdial', 'soft_metab_express', 'soft_freestyle', 'soft_gcimage', 'soft_insight', 'soft_multiq', 'soft_qual', 'soft_skyline', 'soft_sciexos', 'soft_targetlynx', 'soft_tf', ]
software_type = ['Imaging', 'Imaging', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Metabolomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Proteomics', 'Targeted', 'Metabolomics', 'Targeted', 'Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics', 'General', 'Metabolomics', 'Targeted', 'Targeted', 'General', 'Targeted', 'General', 'Targeted', 'Targeted', ]
software_typed_keys = ['%s_%s' % (software_type[i], software_names[i]) for i in range(len(software_keys))]
software_alias_dict = dict(zip(software_keys, software_typed_keys))

# set all non-numeric values to NAN
data_df[software_keys] = data_df[software_keys].apply(pd.to_numeric, errors='coerce')

# set all NAN values to 0
data_df[software_keys] = data_df[software_keys].fillna(0)

# set all values to ing
data_df[software_keys] = data_df[software_keys].astype(int)

# if value is greater than 1, set to 1
data_df[software_keys] = data_df[software_keys].apply(lambda x: np.where(x > 1, 1, x))

data_df['total_software'] = data_df[software_keys].sum(axis=1)

# count maintenance
keys = ['maintenance_warranty', 'maintenance_ful', 'maintenance_prev', 'maintenance_none', ]
data_df['total_maintenance'] = data_df[keys].sum(axis=1)

# reset data_df index startin at 1
data_df.index = data_df.index + 1

CATEGORIES = ['Home'] + key_df['category'].unique().tolist()

# place logo.png
st.sidebar.image('logo.png')

# add selectbox to streamlit sidebar
selection = st.sidebar.selectbox("Select a category", CATEGORIES)

def show_figure(fig):
    # create three columns in 1:2:1 ratio
    col1, col2, col3 = st.columns([1,6,1])
    
    # add plot to col2
    with col2:

        # if fig is string
        if isinstance(fig, str):
            st.code(fig, line_numbers = True)
        else:
            st.pyplot(fig)


def make_simple_count_plot(key):
    fig, ax = plt.subplots()

    # get number of unique values from series
    unique_values = len(data_df[key].unique())

    # set fig size
    fig.set_size_inches(10, 0.8*unique_values)

    # horizontal countplot with black outline
    sns.countplot(data=data_df, y = key, ax=ax, orient='h', palette='Set3', linewidth=1, edgecolor='k')
    show_figure(fig)


def make_simple_bar_chart(key, data = 'a'):
    fig, ax = plt.subplots()

    # if isinstance data is a dataframe, use it, else use data_df
    if isinstance(data, pd.DataFrame):
        data = data
    else:
        data = data_df.copy()

    # sort by key
    data = data.sort_values(by=key, ascending=False)
    # set fig size
    fig.set_size_inches(10, 4)    

    # horizontal countplot with black outline
    try:
        sns.barplot(data=data, x = data_df.index, y = key, ax=ax, orient='v', linewidth=1, edgecolor='k', color='lightgrey')
    except:
        # plot a pandas series
        data.plot.barh(ax=ax, color='lightgrey', linewidth=1, edgecolor='k', width=1)
    show_figure(fig)

def make_text_box(key):
    # find non-null values
    non_null = data_df[data_df[key].notnull()]
    if len(non_null) == 0: return
    text = '\n'.join([row[key] for index, row in non_null.iterrows()])
    show_figure(text)

def make_heatmap_old(keys):
    data = data_df.copy()
    data = data[keys]
    data = data.replace('Checked', 1)
    data = data.replace('Unchecked', 0)
    data['total'] = data.sum(axis=1)
    data = data.sort_values(by='total', ascending=False)
    data = data.drop(columns=['total'])
    data = data.reset_index(drop=True)
    data.index = data.index + 1

    # make plotly histogram with categorical data
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4)
    sns.heatmap(data, ax=ax, cmap='Reds', linewidths=0.1, linecolor='k',yticklabels=True)
    show_figure(fig)

def make_heatmap(keys, aliases = None):
    data = data_df.copy()
    data = data[keys]
    data = data.replace('Checked', 1)
    data = data.replace('Unchecked', 0)

    if aliases is not None:
        data = data.rename(columns=aliases)

    # convert to long format
    data = data.stack().reset_index()

    # rename columns
    data.columns = ['level_0', 'level_1', 'value']

    fig = sns.relplot(
        data=data,
        x="level_0", y="level_1", hue="value", size="value",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=4, sizes=(50, 250), size_norm=(-.2, .8), aspect=2
    )

    # change legend labels
    fig._legend.set_title('Support')
    new_labels = ['No', 'Yes']
    for t, l in zip(fig._legend.texts, new_labels): t.set_text(l)

    # get figure axis
    ax = fig.axes[0][0]
    # set x axis label to 'Lab'
    ax.set_xlabel('Lab')
    # set y axis label to ''
    ax.set_ylabel('')

    # change font size of axis labels and tick labels
    for ax in fig.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), fontsize=18)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)

    # change font size of legend
    for t in fig._legend.texts: t.set_fontsize(16)

    # change font size of title
    fig.ax.set_title(fig.ax.get_title(), fontsize=20)
    show_figure(fig)

def make_staff_breakdown_chart(label, keys, colors):

    fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios':[1,2]})
    data = data_df.copy()

    # remove rows with NAN in staff_total_fte_cal
    data = data[data['staff_total_fte_cal'].notnull()]
    data = data[data['staff_total_fte_cal'] > 0]

    # sort by key
    data = data.sort_values(by='staff_total_fte_cal', ascending=True)
    data = data.reset_index()

    # set fig size
    fig.set_size_inches(10, FACTOR * len(data))
    fig.tight_layout()

    data['staff_total_fte_cal'].plot.barh(ax=axs[0],linewidth=1, edgecolor='k', width = 1, color = 'lightgrey')

    max_satff_fte = data['staff_total_fte_cal'].max()
    axs[0].set_xlim(0, max_satff_fte + 1)
    axs[0].set_title('Total FTE')
    # annotate bars
    for i, v in enumerate(data['staff_total_fte_cal']):
        # if v is a whole number, don't show decimal
        if v.is_integer():
            axs[0].text(v + 0.5, i - 0.3, str(int(v)), color='black')
        else:
            axs[0].text(v + 0.5, i - 0.3, str(v), color='black')

    data = data[keys]
    # strip 'fraction_' from column names
    data.columns = [i.replace('fraction_', '') for i in data.columns]
    bars = data.plot.barh(stacked=True, ax=axs[1],linewidth=1, edgecolor='k', color=colors, width = 1, alpha = 0.5)
    axs[1].set_xlim(0, 100)
    axs[1].set_title(label)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.2, -0.05), ncol=len(data.columns), edgecolor='k')


    for count, rect in enumerate(bars.containers):
        for i, bar in enumerate(rect):
            height = bar.get_height()
            width = bar.get_width()
            y = bar.get_y()
            x = bar.get_x()
            label_text = f'{int(data.iloc[i, count])}' if data.iloc[i, count] > 0 else ''

            #label_text = f'{data.iloc[i, count]:.1f}' if data.iloc[i, count] > 0 else ''
            axs[1].text(x + width / 2, y + height / 2, label_text, ha='center', va='center', fontsize=10)


    # remove x tics and labels
    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    axs[1].set_xticks([])
    axs[1].set_xticklabels([])

    sns.despine(offset=10, trim=True, ax = axs[0], bottom = True)
    sns.despine(offset=10, trim=True, ax = axs[1], bottom = True, left = True)
    
    show_figure(fig)

def make_scatter_chart(x,y, xlabel = '', ylabel = ''):

    fig, ax = plt.subplots()
    data = data_df.copy()

    # remove rows with NAN in staff_total_fte_cal
    data = data[data[x].notnull()]
    data = data[data[x] > 0]

    data.plot.scatter(x=x, y=y, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    show_figure(fig)

def make_stacked_bar_chart(keys, prefix = None, sortby = None, palette = None, legend_at_right = False, alias = None, xlabel='', ylabel=''):

    fig, ax = plt.subplots()
    data = data_df.copy()
    fig.set_size_inches(10, FACTOR * len(data))
    # sort by 'total_instruments'
    if sortby is not None:
        data = data.sort_values(by=sortby, ascending=True)

    data = data[keys]

    if alias is not None:
        data = data.rename(columns=alias)

        # rearrange columns in alphabetical order
        data = data.reindex(sorted(data.columns), axis=1)

    # remove rows with all zeros
    data = data.loc[(data != 0).any(axis=1)]

    # remove any columns with all zeros
    data = data.loc[:, (data != 0).any(axis=0)]

    # strip prefix from column names
    data.columns = [i.replace(prefix, '') for i in data.columns]

    if not palette:
        palette = 'Set3'
        bars = data.plot.barh(stacked=True, ax=ax,linewidth=1, edgecolor='k', width = 1, colormap=palette)
    else:
        num_cols = len(data.columns)
        cmap = plt.cm.get_cmap('rainbow')
        colors = [cmap(i/num_cols) for i in range(num_cols)]
        bars = data.plot.barh(stacked=True, ax=ax,linewidth=1, edgecolor='k', width = 1, color=colors, alpha = 0.5)

    for count, rect in enumerate(bars.containers):
        for i, bar in enumerate(rect):

            height = bar.get_height()
            width = bar.get_width()
            y = bar.get_y()
            x = bar.get_x()

            try:
                label_text = f'{int(data.iloc[i, count])}' if data.iloc[i, count] > 0 else ''
            except:
                label_text = ''
            #label_text = f'{data.iloc[i, count]:.1f}' if data.iloc[i, count] > 0 else ''
            ax.text(x + width / 2, y + height / 2, label_text, ha='center', va='center', fontsize=10)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


    # format legend as two columns to the right of the plot
    if legend_at_right:
        ax.legend(loc='upper center', bbox_to_anchor=(1.23, 1.2), edgecolor='k', ncol=1)
    show_figure(fig)

def make_software_totals_bar_chart(key_dict):
    fig, ax = plt.subplots()
    data = data_df.copy()
    fig.set_size_inches(10, FACTOR * len(data))

    data = data[key_dict.keys()]

    # rename columns using key_dict
    data = data.rename(columns=key_dict)

    # set all non-numeric values to NAN
    data = data.apply(pd.to_numeric, errors='coerce')

    # set all NAN values to 0
    data = data.fillna(0)

    # set all non-zero values to 1
    data[data > 0] = 1
    
    data = data.sum()
    data = data.sort_values(ascending=True)

    data.plot.barh(ax=ax,linewidth=1, edgecolor='k', width = 1, colormap='Set3')
    
    # set x label to 'Number of institutions'
    ax.set_xlabel('Number of Labs')
    
    show_figure(fig)

def make_pricing_model_chart(keys, legend_at_right = False, convert_to_percentage = False, alias = None, subtract = None):

    fig, ax = plt.subplots()
    data = data_df.copy()

    data = data[keys]

    # remove rows with all zeros
    data = data.loc[(data != 0).any(axis=1)]

    # remove rows that only contain nan
    data = data.dropna(how='all')

    # convert to numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    if subtract is not None:
        # replace all NAN values with 0
        data = data.fillna(0)

        # for each value that is not 0 subtract the value in the subtract column
        data[data > 0] = data[data > 0].sub(subtract, axis=0)

        # take the absolute value of each value
        data = data.abs()

    if convert_to_percentage:
        # calculate the percentage of each value for a row
        data = data.div(data.sum(axis=1), axis=0) * 100

    # sort by keys
    data = data.sort_values(by=keys, ascending=True)

    if alias is not None:
        data = data.rename(columns=alias)

    fig.set_size_inches(10, FACTOR * len(data))

    # strip 'fraction_' from column names
    data.columns = [i.replace('pricing_', '') for i in data.columns]
    bars =data.plot.barh(stacked=True, ax=ax,linewidth=1, edgecolor='k', colormap='Set3', width = 1)

    for count, rect in enumerate(bars.containers):
        for i, bar in enumerate(rect):
            height = bar.get_height()
            width = bar.get_width()
            y = bar.get_y()
            x = bar.get_x()
            label_text = f'{int(data.iloc[i, count])}' if data.iloc[i, count] > 0 else ''

            #label_text = f'{data.iloc[i, count]:.1f}' if data.iloc[i, count] > 0 else ''
            ax.text(x + width / 2, y + height / 2, label_text, ha='center', va='center', fontsize=10)

    ax.set_xlim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05), ncol=len(data.columns), edgecolor='k')

    # remove x tics and labels
    ax.set_xticks([])
    ax.set_xticklabels([])

    sns.despine(offset=10, trim=True, ax = ax, bottom = True, left = True)
    if legend_at_right:
        ax.legend(loc='upper center', bbox_to_anchor=(1.23, 0.8), edgecolor='k', ncol=1)
    show_figure(fig)

def make_simple_histogram(keys):

    fig, ax = plt.subplots()
    data = data_df.copy()

    data = data[keys]

    print(data)
    # plot histogram
    data.plot.hist(ax=ax, bins=10, edgecolor='k', linewidth=1, alpha=0.5)
    
    show_figure(fig)

def make_bubble_plot(keys, alias = None):

    d = count_responses(data_df, keys)

    # if alias is not None:
    if alias is not None:
        d = d.rename(columns=alias)

    # convert wide to long
    d = d.stack().reset_index()

    # rename columns
    d.columns = ['Value', 'rates_driver', 'count']

    # convert the value count to int
    d['count'] = d['count'].astype(int)

    fig, ax = plt.subplots()

   # sns.scatterplot(x='Value', y='rates_driver', size='count', data=d, ax=ax, legend='brief', sizes=(100, 2000) , palette='Blues_r')
    sns.scatterplot(
        x='Value', 
        y='rates_driver', 
        size='count', 
        data=d, 
        ax=ax, 
        sizes=(100, 2000),
        hue='count', 
        palette='Blues', 
        legend=False,
        edgecolor='black',
    )

    # add count text inside the spot
    for index, row in d.iterrows():
        ax.text(
            row['Value'], 
            row['rates_driver'], 
            row['count'], 
            color='black', 
            fontsize=10, 
            ha='center', 
            va='center'
        )

     # Set the x-axis tick locator and formatter
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    sns.despine(offset=10, trim=True, ax = ax, bottom = True, left = True)

    # expand all x and y margins by 0.5
    ax.margins(x=0.1, y=0.1)

    ax.text( -0.1, -0.2, 'Primary cost driver', fontsize=12, ha='left', va='center', transform=ax.transAxes )
    ax.text( 0.8, -0.2, 'Minor cost driver', fontsize=12, ha='left', va='center', transform=ax.transAxes )
    show_figure(fig)



def count_responses(dataframe, selected_options):
    """
    Counts the number of responses to selected options (columns) in a DataFrame.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame with responses.
        selected_options (list): List of column names to count responses for.

    Returns:
        pandas.DataFrame: DataFrame with counts of responses to each selected option.
    """
    return dataframe[selected_options].apply(pd.Series.value_counts).fillna(0).astype(int)

if selection == 'Overview':
    st.header('Overview')
    st.markdown('### Host institution type')
    make_simple_count_plot('host')
    st.markdown("""---""")

    st.markdown('### Facility type')
    make_simple_count_plot('facility_type')
    st.write('Other facility types were')
    make_text_box('facitity_type_other')
    st.markdown("""---""")

    st.markdown('### Position of survey respondent')
    make_simple_count_plot('position')
    st.markdown("""---""")

    st.markdown('### Accreditation Status')
    make_simple_count_plot('accreditation')
    st.markdown("""---""")

    st.markdown('### Support from the host institution')

    supports = ['Invoicing', 'Ordering', 'Marketing (inc. websites)', 'Fundraising', 'Grant-writing support', 'IT', 'Workshop', 'Other']
    keys = ['support___%s' % i for i in range(1, 9)]
    key_dict = dict(zip(keys, supports))
    make_heatmap(keys, aliases = key_dict)
    st.write('Other support was')
    make_text_box('host_other_2')


elif selection == 'Staffing':
    st.header('Staffing')

    st.markdown('### Total FTE')
    make_simple_bar_chart('staff_total_fte_cal')
    st.markdown("""---""")

    st.markdown('### Staff Sex Breakdown')
    make_staff_breakdown_chart('Staff Sex', ['fraction_female', 'fraction_male'], ['pink', 'blue'])
    st.markdown("""---""")

    st.markdown('### Staff Role Breakdown')
    make_staff_breakdown_chart('Staff Role', ['fraction_academic', 'fraction_professional', 'fraction_administrative'], ['red', 'orange', 'green'])
    st.markdown("""---""")

    st.markdown('### Staff Contract Type Breakdown')
    make_staff_breakdown_chart('Staff Contract', ['fraction_continuing', 'fraction_fixed_term', 'fraction_casual'], ['purple', 'yellow', 'blue'])
    st.markdown("""---""")

    st.markdown('### Staff Cost Breakdown')
    make_scatter_chart('staff_total_fte_cal', 'staff_cost_aud', 'Total FTE', 'Total Cost (AUD)')
    st.markdown("""---""")

elif selection == 'Instrumentation':
    st.header('Instrumentation')

    st.markdown('### Total instruments')
    make_simple_bar_chart('total_instruments')
    st.markdown("""---""")

    st.markdown('### Total instruments by type')
    make_stacked_bar_chart(['total_instrument_count' + i for i in instruments], sortby = 'total_instruments', prefix = 'total_instrument_countno_ms_', palette = 'rainbow')
    st.write('Other instruments were')
    make_text_box('ms_other_type')
    st.markdown("""---""")

    st.markdown('### Total instruments by type')
    cols = ['total_' + i for i in instruments]
    # select only cols from data_df
    data = data_df[cols]
    #strip 'total_no_ms_' from column names
    data.columns = [i.replace('total_no_ms_', '').upper() for i in data.columns]
    # sum cols
    data = data.sum(axis=0)
    # convert to dataframe
    data = pd.DataFrame(data, columns=['total'])
    make_simple_bar_chart('total', data = data)

    st.markdown('### Staff FTE per instrument')
    make_simple_bar_chart('staff_fte_per_instrument')

    keys = ['maintenance_warranty','maintenance_ful','maintenance_prev','maintenance_none']
    values = ['Warranty', 'Full', 'Preventative', 'None']
    alias_dict = dict(zip(keys, values))

    st.markdown('### Maintenance')
    make_stacked_bar_chart(keys, '',alias = alias_dict, sortby = 'total_maintenance')

    st.write('Other maintenance comments were')
    make_text_box('maintenance_other')

    st.markdown('### Instrument funding')
    keys = [ 'ms_fund_source_host', 'ms_fund_source_phil', 'ms_fund_source_state', 'ms_fund_source_fed', 'ms_fund_source_riip', 'ms_fund_source_charity', 'ms_fund_source_self', 'ms_fund_source_other']
    values = ['Host institution', 'Philanthropic', 'State', 'Federal (ex. NCRIS/RIIP)', 'NCRIS/RIIP', 'Charity', 'Self', 'Other']
    alias_dict = dict(zip(keys, values))
    make_pricing_model_chart(keys, alias = alias_dict, convert_to_percentage=True, legend_at_right=True)

elif selection == 'Software':

    st.header('Software')
    st.markdown('### Number of labs that use each software package')
    # create a dict of software names and keys
    software_dict = dict(zip(software_keys, software_names))
    make_software_totals_bar_chart(software_dict)
    st.write('Other software was')
    make_text_box('soft_other')
    st.markdown("""---""")

    st.markdown('### Lab use of different software packages')
    make_stacked_bar_chart(software_keys, sortby = 'total_software', prefix = 'soft_', palette = 'rainbow', legend_at_right=True, alias = software_alias_dict, ylabel = 'Lab')
    st.markdown("""---""")

elif selection == 'Pricing':

    st.header('Pricing')

    st.markdown('### What billing model does your lab use?')
    keys = [_ for _ in data_df.columns if 'pricing_' in _]
    make_pricing_model_chart(keys, legend_at_right = True)
    st.markdown("""---""")

    st.markdown('### Which mechanism best matches how you sety our internal rates?')
    keys = [_ for _ in data_df.columns if 'rates_driver' in _]
    values = ['Maximum profit', 'What the market will bare', 'Cover direct costs', 'What the host will subsidise', ]
    rates_dict = dict(zip(keys, values))
    make_bubble_plot(keys, alias = rates_dict)

    st.markdown('### What pricing rates do you have?')
    keys = [_ for _ in data_df.columns if 'rates___' in _]
    values = ['Academia (internal)', 'Academia (external)', 'Commercial customer', 'Other']  
    rates_dict = dict(zip(keys, values))
    make_heatmap(keys, aliases = rates_dict)

    st.markdown('### What is the average markup for external academics? (in %)')
    make_simple_histogram('rates_ac_int_to_ac_ex')
    st.markdown("""---""")

    st.markdown('### What is the average markup for commercial customers? (in %)')
    make_simple_histogram('rates_ac_int_to_com')
    st.markdown("""---""")

elif selection == 'Home':

    _, col2, _ = st.columns([1, 8, 1])
    with col2:
        st.header('2022 Australasian Core Facility Member Survey Results')
        st.write('A big THANK YOU to all those who participated in the 2022 ACF member survey!\
                The survey covered a wide range of topics, including staffing, instrumentation, pricing, funding and support from host institutions.\
                We are thrilled to present the results here. These results are a snapshot of the current state of the Australian and New Zealand core facility landscape. We hope you find them useful and informative.')
        
        st.markdown('### Highlights')

        col1, col2 = st.columns([1,10])
        with col2:
            st.write(f'* {len(data_df)} facilities responded to the survey')
            st.write(f'* ACF parter labs employ {int(data_df["staff_total_fte_cal"].sum())} FTE staff')
            st.write(f'* ACF partner labs have {data_df["total_instruments"].sum()} instruments')

        st.markdown('### Raw data')
        st.write('The raw data from the survey is available for download. Note that this is the file that was directly produced by REDCap. Manual cleanup was required to produce the final results here.')
        with open(DATA_FILE, "rb") as file:
            btn = st.download_button(
                    label="Download",
                    data=file,
                    file_name="2022_ACF_member_survey_results.csv",
                    mime="text/csv",
                )

        st.markdown('### Acknowledgements')
        st.write('The Australasian Core Facility Association (ACFA) would like to thank the following people for their contributions to the 2022 ACF member survey:')

        acks = {
            'Survey Coordination': ['Ben Crossett (USyd)', 'Ralf Schittenhelm (Monash)'],
            'Survey Design': ['Paula Burton (Mass Dynamics)', 'Mark Condina (UniSA)'],
            'Survey Testing': ['Matt Padula (UTS)', 'Tara Pukala (UoA)', 'Nick Williamson (Unimelb)'],
            'Survey Analysis': ['Naveed Nadiv (USyd)', 'Michael Leeming (Unimelb)']
        }

        cols = st.columns(2)
        counter = 0
        for k,v in acks.items():
            if counter > 1: counter = 0
            with cols[counter]:
                counter += 1
                st.markdown('###### ' + k)
                st.markdown('* ' + ', '.join(v))

