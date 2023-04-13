import os, sys
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ACF survey",layout='wide')

print('\n\n\n')
WD = os.getcwd()
DATA_DIR = os.path.join(WD, 'data')

KEY_FILE = os.path.join(DATA_DIR, 'key_2.csv')
DATA_FILE = os.path.join(DATA_DIR, 'data.csv')

key_df = pd.read_csv(KEY_FILE, sep ='\t')
data_df = pd.read_csv(DATA_FILE)

CATEGORIES = key_df['category'].unique()

# add streamlit sidebar
st.sidebar.write("Hello World")

# add selectbox to streamlit sidebar
selection = st.sidebar.selectbox("Select a category", CATEGORIES)

# subset key_df to selection
key_df = key_df[key_df['category'] == selection]

# remove rows from key_df where group is null
key_df = key_df[key_df['group'].notnull()]

# get unique groups
groups = key_df['group'].unique()

for group in groups:
    st.header(group)
    q_group_df = key_df[key_df['group'] == group]
    fields = q_group_df['field'].unique()

    for field in fields:
        field_df = q_group_df[q_group_df['field'] == field]

        if len(field_df) == 1:
            redcap = field_df['redcap'].values[0]
            data = data_df[redcap].to_frame(name=redcap)
        else:
            redcap_fields = field_df['redcap'].values
            data = data_df[redcap_fields]
            
            data = data.replace('Checked', 1)
            data = data.replace('Unchecked', 0)

            # convert wide to long
          #  data = pd.melt(data, var_name='support', value_name='value')

        presentation_type = field_df['presentation_type'].values[0]

        if presentation_type == 'text':
            # find non-null values
            non_null = data[data[redcap].notnull()]
            if len(non_null) == 0: continue
            text = '\n'.join([row[0] for index, row in non_null.iterrows()])
            st.write(field)
            st.code(text)

        if presentation_type == 'histogram':
            # make plotly histogram with categorical data
            fig = px.histogram(data, y=redcap, color=redcap, opacity=0.5)
            fig.update_layout(height=300)
            st.write(field)
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)

        if presentation_type == 'heatmap':
            data['total'] = data.sum(axis=1)
            data = data.sort_values(by='total', ascending=False)
            data = data.drop(columns=['total'])
            data = data.reset_index(drop=True)

            fig = px.imshow(data, color_continuous_scale='Reds')
            st.write(field)
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        
        if presentation_type == 'scatter':
            # sort by redcap field
            data = data.sort_values(by=redcap, ascending=True)
            data['index'] = list(range(len(data)))
            fig = px.scatter( data, x="index", y=redcap, color_continuous_scale="reds" )
            fig.update_layout(height=400)
            st.write(field)
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        
    st.markdown("""---""") 

