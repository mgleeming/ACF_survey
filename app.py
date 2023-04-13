import os, sys
import pandas as pd
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


st.set_page_config(page_title="ACF survey",layout='wide')

print('==========================================================')
WD = os.getcwd()
DATA_DIR = os.path.join(WD, 'data')

KEY_FILE = os.path.join(DATA_DIR, 'key_2.csv')
DATA_FILE = os.path.join(DATA_DIR, 'data.csv')

key_df = pd.read_csv(KEY_FILE, sep ='\t')
data_df = pd.read_csv(DATA_FILE)
print(list(key_df))
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

def make_simple_count_plot(key):
    fig, ax = plt.subplots()

    # get number of unique values from series
    unique_values = len(data_df[key].unique())

    # set fig size
    fig.set_size_inches(10, 0.8*unique_values)

    # horizontal countplot with black outline
    sns.countplot(data=data_df, y = key, ax=ax, orient='h', palette='Set3', linewidth=1, edgecolor='k')
    st.pyplot(fig)
    return

def make_simple_bar_chart(key):
    fig, ax = plt.subplots()

    data = data_df.copy()
    # sort by key
    data = data.sort_values(by=key, ascending=False)
    # set fig size
    fig.set_size_inches(10, 4)

    # horizontal countplot with black outline
    sns.barplot(data=data, x = data_df.index, y = key, ax=ax, orient='v', linewidth=1, edgecolor='k')
    st.pyplot(fig)
    return

def make_text_box(key):
    # find non-null values
    non_null = data_df[data_df[key].notnull()]
    if len(non_null) == 0: return
    print(non_null)
    text = '\n'.join([row[key] for index, row in non_null.iterrows()])
    st.code(text)
    return

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
    st.pyplot(fig)


def make_heatmap(keys):
    data = data_df.copy()
    data = data[keys]
    data = data.replace('Checked', 1)
    data = data.replace('Unchecked', 0)

    # convert to long format
    data = data.stack().reset_index()

    # rename columns
    data.columns = ['level_0', 'level_1', 'value']

    fig = sns.relplot(
        data=data,
        x="level_0", y="level_1", hue="value", size="value",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=8, sizes=(50, 250), size_norm=(-.2, .8),
    )

    # change legend labels
    fig._legend.set_title('Support')
    new_labels = ['No', 'Yes']
    for t, l in zip(fig._legend.texts, new_labels): t.set_text(l)

    st.pyplot(fig)

# reset data_df index startin at 1
data_df.index = data_df.index + 1

CATEGORIES = key_df['category'].unique()

# add streamlit sidebar
st.sidebar.write("Hello World")

# add selectbox to streamlit sidebar
selection = st.sidebar.selectbox("Select a category", CATEGORIES)

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
    make_heatmap(['support___%s' % i for i in range(1, 9)])

elif selection == 'Staffing':
    st.header('Staffing')
    st.markdown('### Total FTE')
    make_simple_bar_chart('staff_total_fte_cal')
    st.markdown("""---""")

    st.markdown('### Staff Sex Breakdown')
    
    # create two subplots in a 2:1 ratio
    fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios':[1,2]})
    data = data_df.copy()

    # remove rows with NAN in staff_total_fte_cal
    data = data[data['staff_total_fte_cal'].notnull()]

    # sort by key
    data = data.sort_values(by='staff_total_fte_cal', ascending=True)
    data = data.reset_index()
    # set fig size
    fig.set_size_inches(10, 0.2* len(data))
    fig.tight_layout()

    data['staff_total_fte_cal'].plot.barh(ax=axs[0],linewidth=1, edgecolor='k', width = 1, color = 'lightgrey')
    data = data[['fraction_female', 'fraction_male']]
    data.plot.barh(stacked=True, ax=axs[1],linewidth=1, edgecolor='k', color=['pink', 'blue'], width = 1)

    legend = axs[1].get_legend()
    legend.set_bbox_to_anchor((1, 1))
    st.pyplot(fig)
    st.markdown("""---""")

    st.markdown('### Staff Role Breakdown')

    # create two subplots in a 2:1 ratio
    fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios':[1,2]})
    data = data_df.copy()

    # remove rows with NAN in staff_total_fte_cal
    data = data[data['staff_total_fte_cal'].notnull()]

    # sort by key
    data = data.sort_values(by='staff_total_fte_cal', ascending=True)
    data = data.reset_index()
    data['staff_total_fte_cal'].plot.barh(ax=axs[0],linewidth=1, edgecolor='k', width = 1, color = 'lightgrey')
    data = data[['fraction_academic', 'fraction_professional', 'fraction_administrative']]
    data.plot.barh(stacked=True, ax=axs[1],linewidth=1, edgecolor='k', color=['red', 'orange', 'green'], width = 1)

    legend = axs[1].get_legend()
    legend.set_bbox_to_anchor((1, 1))
    st.pyplot(fig)
    st.markdown("""---""")

    st.markdown('### Staff Contract Type Breakdown')

    # create two subplots in a 2:1 ratio
    fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios':[1,2]})
    data = data_df.copy()

    # remove rows with NAN in staff_total_fte_cal
    data = data[data['staff_total_fte_cal'].notnull()]

    # sort by key
    data = data.sort_values(by='staff_total_fte_cal', ascending=True)
    data = data.reset_index()
    data['staff_total_fte_cal'].plot.barh(ax=axs[0],linewidth=1, edgecolor='k', width = 1, color = 'lightgrey')
    data = data[['fraction_continuing', 'fraction_fixed_term', 'fraction_casual']]
    data.plot.barh(stacked=True, ax=axs[1],linewidth=1, edgecolor='k', color=['red', 'orange', 'green'], width = 1)

    legend = axs[1].get_legend()
    legend.set_bbox_to_anchor((1, 1))
    st.pyplot(fig)
    st.markdown("""---""")