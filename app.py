# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:49:40 2021
 
@author: Administrator
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_validate
import plotly.express as px
import pickle
import shap
import streamlit.components.v1 as components
from interpret import show
from interpret.blackbox import ShapKernel
st.image('https://www.anaf.fr/wp-content/uploads/2020/09/OpenClassroom_LOGO.png', width=800)

st.header('Home Credit Default Risk Dashboard')

def histogram(df, x='str', legend=True, client=None): 
    '''client = [df_test, input_client] '''
    if x == "TARGET":
        fig = px.histogram(df,
                        x=x,
                        color="TARGET",
                        width=300,
                        height=200,
                        category_orders={"TARGET": [1, 0]})
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=50))
    else:
        fig = px.histogram(df,
                x=x,
                color="TARGET",
                width=300,
                height=200,
                category_orders={"TARGET": [1, 0]},
                barmode="group",
                histnorm='percent')
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    if legend == True:
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
    else:
        fig.update_layout(showlegend=False)
    if client: 
        client_data = client[0][client[0].SK_ID_CURR ==  client[1]]
        vline = client_data[x].to_numpy()[0]
        print(vline)
        
        fig.add_vline(x=vline, line_width=3, line_dash="dash", line_color="black")
    return fig  



# data import

df_train=pd.read_csv("https://raw.githubusercontent.com/housnaki/septiemeprojet/main/train_Xy_sample.csv",encoding="utf-8")
df_test = pd.read_csv("https://raw.githubusercontent.com/housnaki/septiemeprojet/main/X_test_sample.csv",encoding="utf-8")
df_description = pd.read_csv("https://raw.githubusercontent.com/housnaki/septiemeprojet/main/HomeCredit_columns_description.csv",encoding='cp1252')
#FILENAME_MODEL = "mymodel1.pkl"
load_clf = pickle.load(open('mymodel1.pkl', 'rb'))
sb = st.sidebar # add a side bar

sb.image('https://user.oc-static.com/upload/2019/02/25/15510866018677_logo%20projet%20fintech.png', width=280)

sb.markdown('**Choose a client ID:**')
np.random.seed(12) # one major change is that client is directly asked as input since sidebar
label_test = df_test['SK_ID_CURR'].sample(200).sort_values()
radio = sb.radio('', ['Random client ID', 'Type client ID'])

if radio == 'Random client ID': # Choice choose preselected seed13 or a known client ID
    input_client = sb.selectbox('Select random client ID', label_test)
if radio == 'Type client ID':
    input_client = int(sb.text_input('Type client ID', value=113798))

sb.markdown('**Navigation**')
rad = sb.radio('', [' Home', 
' Client data',
'Client prediction'])

# defining containers of the app
header = st.container()
dataset = st.container()
eda = st.container()
model_training = st.container()
model_predict = st.container()

from PIL import Image

#image1 = Image.open('/app/project_7_oc_dashboard/Global_feature_importance.png')


if rad == ' Home': # with this we choose which container to display on the screen
    with eda: 

        
        st.subheader("Here's the test dataframe.")
        max_row = st.slider("Select at many row you wanna visualize", value=1000, min_value=1, max_value=len(df_test)) 
        st.write(df_test.head(max_row)) 

        st.subheader("Here's the dataframe_columns_description.")
        max_row = st.slider("Select at many row you wanna visualize", value=218, min_value=1, max_value=len(df_description)) 
        st.write(df_description.head(max_row))
        
        
        st.subheader("Here's the global features importance of data.")        
        st.image("https://github.com/housnaki/septiemeprojet/blob/main/Global_feature_importance.png", width=700)
        
        st.header("**Overview of exploratory data analysis.** \n ----")
        st.subheader("Plotting distributions of target and some features.")
 
        
        col1, col2, col3 = st.columns(3) # 3 cols with histogram = home-made func
        col1.plotly_chart(histogram(df_train, x='TARGET'), use_container_width=True)
        col2.plotly_chart(histogram(df_train, x='CODE_GENDER'), use_container_width=True)
        col3.plotly_chart(histogram(df_train, x='PAYMENT_RATE'), use_container_width=True)       

        col1, col2, col3 = st.columns(3) # 3 cols with histogram = home-made func
        col1.plotly_chart(histogram(df_train, x='EXT_SOURCE_1'), use_container_width=True)
        col2.plotly_chart(histogram(df_train, x='EXT_SOURCE_2'), use_container_width=True)
        col3.plotly_chart(histogram(df_train, x='EXT_SOURCE_3'), use_container_width=True)       

        col1, col2, col3 = st.columns(3) # 3 cols with histogram = home-made func
        col1.plotly_chart(histogram(df_train, x='DAYS_BIRTH'), use_container_width=True)
        col2.plotly_chart(histogram(df_train, x='DAYS_EMPLOYED'), use_container_width=True)
        col3.plotly_chart(histogram(df_train, x='AMT_ANNUITY'), use_container_width=True)       

if rad == ' Client data':  
    with eda:
        st.header("**Client's data.** \n ----")
        # retrieving whole row of client from sidebar input ID
        client_data = df_test[df_test.SK_ID_CURR == input_client]
        #client_data = client_data.drop(['SK_ID_CURR'])  

        st.subheader(f"**Client ID: {input_client}.**")
        # plotting features from train set, with client's data as dashed line (client!=None in func)
        st.subheader("Ranking client in some features.")      
        col1, col2, col3 = st.columns(3)
        col1.plotly_chart(histogram(df_train, x='EXT_SOURCE_1', client=[df_test, input_client]), use_container_width=True)
        col2.plotly_chart(histogram(df_train, x='EXT_SOURCE_2', client=[df_test, input_client]), use_container_width=True)
        col3.plotly_chart(histogram(df_train, x='EXT_SOURCE_3', client=[df_test, input_client]), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.plotly_chart(histogram(df_train, x='DAYS_BIRTH', client=[df_test, input_client]), use_container_width=True)
        col2.plotly_chart(histogram(df_train, x='DAYS_EMPLOYED', client=[df_test, input_client]), use_container_width=True)
        col3.plotly_chart(histogram(df_train, x='AMT_ANNUITY', client=[df_test, input_client]), use_container_width=True)
       
        st.subheader("More information about this client.")
        # st.subheader("More information about this client.")
        # displaying values from a dropdown (had issues with NaNs that's why .dropna())

        info = st.selectbox('What info?', client_data.columns.sort_values())     
        info_print = client_data[info].to_numpy()[0]

        st.subheader(info_print)
        # displaying whole non NaNs row
        st.write("All client's data.")
        st.write(client_data)


#modelEssaiP7= 'finalized_model.sav'
client_data = df_test[df_test.SK_ID_CURR == input_client]


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

df_test2=df_test.drop(["SK_ID_CURR"],1)

feat_list= df_test2.select_dtypes(include=np.number)

if rad == 'Client prediction': 
    with model_predict:

        col1, col2 = st.columns(2)
        col1.markdown(f'**Client ID: {input_client}**')

        if col2.button('Predict & plot!'):

            try: 
                model = pickle.load(open(FILENAME_MODEL, 'rb'))
            except:
                raise 'You must train the model first.'
            # finding client row index in testset
            idx = df_test.SK_ID_CURR[df_test.SK_ID_CURR == input_client].index
            client = df_test2.iloc[idx, :]
            y_prob = model.predict_proba(client)

             
            if (y_prob).T[1] < (y_prob.T)[0]:
                st.subheader(f"**Probability of successful payment.**")
            else:
                st.subheader(f"**Probability of Failure payment.**")
            # plotting pie plot for proba, finding good h x w was a bit tough
            fig = px.pie(values=(pd.DataFrame(y_prob)).loc[0], names=["0","1"])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"**SHAP explanation force plots for the client.**")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(client)
            shap.initjs()
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], client))

            st.subheader(f"**SHAP explanation summary plots for the client.**")


            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.summary_plot(shap_values[1], client),showPyplotGlobalUse = False)

sb.markdown('**By Housna KOUIDRI**') 
