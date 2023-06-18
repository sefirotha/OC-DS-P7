"""Application : Dashboard de Crédit Score

Auteur: Erwan Berthaud
Source: 
Local URL: 
Network URL: 
"""
# ====================================================================
# Version : 1.0 -
# ====================================================================

__version__ = '1.0.0'


import streamlit as st
from PIL import Image
import requests
import pickle
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import shap
import json


# ====================================================================
# VARIABLES STATIQUES
# ====================================================================
# Répertoire de sauvegarde du meilleur modèle
# best_model = "./Data/Model/tuned_lgbm_f10.pkl"
# Test set
file_test_set = "./Data/Processed_data/test_df_LFS.pkl"
# Client info, raw
file_client_test = "./Data/Processed_data/application_test_LFS.pkl"
# Train set
file_train_set = "./Data/Processed_data/train_df_LFS.pkl"
# Shap values
shap_values_set = "./Data/Processed_data/230616_shap_values_LFS.pickle"

# ====================================================================
# IMAGES
# ====================================================================
# Logo de l"entreprise
logo = Image.open("./Data/images/logo.png")


# ====================================================================
# FONCTIONS
# ====================================================================
@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"]/365), 2)
    return data_age



# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header = """
    <head>
        <title>Application Dashboard credit Score</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Home Crédit Group, Dashboard, prêt, crédit score">
        <meta name="description" content="Application de Crédit Score - dashboard">
        <meta name="author" content="Erwan Berthaud">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:300%; color:#0031CC; font-family:Arial"> Prêt à dépenser <br>
        <h2 style="color:GREY; font-family:Georgia"> DASHBOARD</h2>
        <hr style= "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1.5px;"/>
     </h1>
"""
st.set_page_config(page_title="Prêt à dépenser - Dashboard",
                   page_icon="", layout="wide")
st.markdown(
    "<style>body{background-color: #fbfff0}</style>", unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

# ====================================================================
# CHARGEMENT DES DONNEES
# ====================================================================


# Chargement du modèle et des différents dataframes
# Optimisation en conservant les données non modifiées en cache mémoire
# @st.cache(persist = True)
def load():
    with st.spinner("Data import"):

        # Import du dataframe du test set nettoyé et pré-procédé
        with open(file_test_set, "rb") as df_test_set:
            df_test_set = pickle.load(df_test_set)
            df_test_set.drop("TARGET", axis=1, inplace=True)

        # Import du dataframe du train set nettoyé et pré-procédé
        with open(file_train_set, "rb") as df_train_set:
            df_train_set = pickle.load(df_train_set)

        # Import du fichier application test original pour les infos clients
        with open(file_client_test, "rb") as df_client_test:
            df_client_test = pickle.load(df_client_test)

        # Import du dataframe du test set brut original
        with open(shap_values_set, 'rb') as shap_values_array:
            shap_values = pickle.load(shap_values_array)

    return df_test_set, df_client_test, df_train_set, shap_values


# Chargement des dataframes et du modèle
df_test_set, df_client_test, df_train_set, shap_values = load()

df_info_client = df_client_test[["SK_ID_CURR",
                                 "DAYS_BIRTH",
                                 "CODE_GENDER",
                                 "NAME_FAMILY_STATUS",
                                 "CNT_CHILDREN",
                                 "NAME_EDUCATION_TYPE",
                                 "NAME_INCOME_TYPE",
                                 "DAYS_EMPLOYED",
                                 "AMT_INCOME_TOTAL",]].copy()

df_info_client["AGE"] = - \
    np.round(df_info_client["DAYS_BIRTH"] / 365, 0).astype(int)
df_info_client["YEARS EMPLOYED"] = - \
    np.round(df_info_client["DAYS_EMPLOYED"] / 365, 0).astype(int)
df_info_client = df_info_client[["SK_ID_CURR",
                                 "AGE",
                                 "CODE_GENDER",
                                 "NAME_FAMILY_STATUS",
                                 "CNT_CHILDREN",
                                 "NAME_EDUCATION_TYPE",
                                 "NAME_INCOME_TYPE",
                                 "YEARS EMPLOYED",
                                 "AMT_INCOME_TOTAL",]]

df_info_client.rename(columns={"CODE_GENDER": "GENDER",
                               "NAME_FAMILY_STATUS": "FAMILY STATUS",
                               "CNT_CHILDREN": "NB OF CHILDREN",
                               "NAME_EDUCATION_TYPE": "EDUCATION",
                               "NAME_INCOME_TYPE": "INCOME SOURCE",
                               "AMT_INCOME_TOTAL": "INCOME",
                               },
                      inplace=True)

df_info_pret = df_client_test[["SK_ID_CURR",
                               "NAME_CONTRACT_TYPE",
                               "AMT_CREDIT",
                               "AMT_ANNUITY",
                               "AMT_GOODS_PRICE",
                               "NAME_HOUSING_TYPE",]].copy()

df_info_pret.rename(columns={"NAME_CONTRACT_TYPE": "CONTRACT TYPE",
                             "AMT_CREDIT": "AMOUNT REQUESTED ($)",
                             "AMT_ANNUITY": "ANNUITY ($)",
                             "AMT_GOODS_PRICE": "GOODS' PRICE ($)",
                             "NAME_HOUSING_TYPE": "HOUSING TYPE",
                             },
                    inplace=True)

# ====================================================================
# CHOIX DU CLIENT
# ====================================================================

html_select_client = """
    <div class="card">
      <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #B8C8FA; padding-top: 5px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color:#B8C8FA; color:#0031CC;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          Client information / Loan request info
        </h3>
      </div>
    </div>
    """

st.markdown(html_select_client, unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("")
        col1.header("**ID Client**")
        client_id = col1.selectbox("Select a client :",
                                   df_test_set["SK_ID_CURR"].unique())
    with col2:
        # Infos principales client
        st.write("*Client data*")
        client_info = df_info_client.loc[df_info_client["SK_ID_CURR"] == client_id]
        client_info.set_index("SK_ID_CURR", inplace=True)
        st.dataframe(client_info.style.format(precision = 0))
        # Infos principales sur la demande de prêt
        # st.write("*Demande de prêt*")
        client_pret = df_info_pret[df_info_pret["SK_ID_CURR"]
                                   == client_id].iloc[:, :]
        client_pret.set_index("SK_ID_CURR", inplace=True)
        st.dataframe(client_pret.style.format(precision = 0))

# ====================================================================
# SCORE - PREDICTIONS
# ====================================================================

html_score = """
    <div class="card">
      <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #B8C8FA; padding-top: 5px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color:#B8C8FA; color:#0031CC;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          Probability of default
        </h3>
      </div>
    </div>
    """

st.markdown(html_score, unsafe_allow_html=True)


# Préparation des données à afficher dans la jauge ==============================================

# ============== Score du client en pourcentage ==> en utilisant le modèle ======================

# model_data = joblib.load(best_model)
# # Sélection des variables du client étudié
X_test = df_test_set[df_test_set["SK_ID_CURR"] == client_id]
# # Prédictions de probabiltés

data = X_test.squeeze().to_dict()


response_proba = requests.post(
    "https://credit-score-api-7acf30a43d8a.herokuapp.com/predict_proba", json=data)
y_proba = json.loads(response_proba.text)["prediction_proba"]
score_client = int(np.rint(y_proba * 100))

response_label = requests.post("https://credit-score-api-7acf30a43d8a.herokuapp.com/predict", json=data)
client_label = int(json.loads(response_label.text)["prediction"])


# Graphique de jauge du cédit score ==========================================
fig_jauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    # Score du client
    value=score_client,
    domain={"x": [0, 1], "y": [0, 1]},
    title={"text": "Risk probablity of default", "font": {"size": 24}},
    gauge={"axis": {"range": [None, 100],
                    "tickwidth": 3,
                    "tickcolor": "darkblue"},
           "bar": {"color": "white", "thickness": 0.25},
           "bgcolor": "white",
           "borderwidth": 2,
           "bordercolor": "gray",
           "steps": [{"range": [0, 25], "color": "Green"},
                     {"range": [25, 49.49], "color": "LimeGreen"},
                     {"range": [49.5, 50.5], "color": "red"},
                     {"range": [50.51, 75], "color": "Orange"},
                     {"range": [75, 100], "color": "Crimson"}],
           "threshold": {"line": {"color": "white", "width": 10},
                         "thickness": 0.8,
                         "value": score_client}}))

fig_jauge.update_layout(paper_bgcolor="white",
                        height=400, width=500,
                        font={"color": "darkblue", "family": "Arial"},
                        margin=dict(l=0, r=0, b=0, t=0, pad=0))

with st.container():
    # JAUGE + récapitulatif du score moyen des voisins
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.plotly_chart(fig_jauge)
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        # Texte d"accompagnement de la jauge
        if 0 <= score_client < 25:
            score_text = "Probability of default : LOW"
            st.success(score_text)
        elif 25 <= score_client < 50:
            score_text = "Probability of default: MIDDLE LOW"
            st.success(score_text)
        elif 50 <= score_client < 75:
            score_text = "Probability of default : MIDDLE HIGH"
            st.warning(score_text)
        else:
            score_text = "Probability of default: HIGH"
            st.error(score_text)
        st.write("")
        if client_label == 0:
            credit_label = "Credit request accorded"
            st.success(credit_label)

        else:
            credit_label = "Credit request denied"
            st.error(credit_label)


# ====================================================================
# SIDEBAR
# ====================================================================

# Toutes Les informations non modifiées du client courant
df_client_origin = df_client_test[df_client_test['SK_ID_CURR'] == client_id]
df_client_origin.set_index("SK_ID_CURR", inplace=True)
# Les informations pré-procédées du client courant
df_current_client = df_test_set[df_test_set['SK_ID_CURR'] == client_id]
df_current_client.set_index("SK_ID_CURR", inplace=True)


# --------------------------------------------------------------------
# LOGO
# --------------------------------------------------------------------
# Chargement du logo de l'entreprise
st.sidebar.image(logo, width=240, caption=" Dashboard - Decision support tool",
                 use_column_width='always')

# --------------------------------------------------------------------
# PLUS INFORMATIONS
# --------------------------------------------------------------------


def all_infos_clients():
    ''' Affiche toutes les informations sur le client courant
    '''
    html_all_infos_clients = """
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #B8C8FA; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#B8C8FA; color:#0031CC;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Plus infos
                  </h3>
            </div>
        </div>
        """

    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES ===========================
    if st.sidebar.checkbox("Display all info on current client ?"):

        st.markdown(html_all_infos_clients, unsafe_allow_html=True)

        with st.spinner('**Display all info on the current client...**'):

            with st.expander('All info, current client',
                             expanded=True):
                st.dataframe(df_client_origin.style.format(precision = 0))
                st.dataframe(df_current_client.style.format(precision = 0))


st.sidebar.subheader('More info')
all_infos_clients()


# --------------------------------------------------------------------
# CLIENTS SIMILAIRES 
# --------------------------------------------------------------------
def infos_clients_similaires():
    ''' Affiche les informations sur les clients similaires :
            - traits stricts.
            - demande de prêt
    '''
    html_clients_similaires="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #B8C8FA; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#B8C8FA; color:#0031CC;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Clients similaires
                  </h3>
            </div>
        </div>
        """
    titre = True
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Compare with other clients ?"):     
        
        if titre:
            st.markdown(html_clients_similaires, unsafe_allow_html=True)
            titre = False

        with st.spinner('**Printing comparing plots...**'):                 
                       
            with st.expander('Comparison current client / other clients',
                             expanded=True):
                st.write("")
                st.write("")
                st.write("")
                with st.container():
                    col1, col2 = st.columns([1.5, 1])
                    with col1:
                        # ============ Histogramme de l'âge des clients ==============================================
                        age_client = df_info_client['AGE'].loc[df_info_client['SK_ID_CURR']== client_id]
                        fig_age = px.histogram(df_info_client, x="AGE", color="GENDER", hover_data=df_info_client.columns)
                        fig_age.add_vline(x=int(age_client), line_dash = 'dash', line_color = 'firebrick')
                        fig_age.update_layout()
                        fig_age.update_layout(paper_bgcolor="white",
                        height=400, 
                        width=500,
                        font={"color": "darkblue", "family": "Arial"},
                        margin=dict(l=20, r=20, b=20, t=20, pad=10),
                        title_text='Distribution of age')
                        st.plotly_chart(fig_age)

                        # ============ Histogramme du revenu des clients ==============================================
                        revenu_client = df_info_client['INCOME'].loc[df_info_client['SK_ID_CURR']== client_id]
                        fig_income = px.histogram(df_info_client, x="INCOME", hover_data=df_info_client.columns)
                        fig_income.add_vline(x=int(revenu_client), line_dash = 'dash', line_color = 'firebrick')
                        fig_income.update_layout()
                        fig_income.update_layout(paper_bgcolor="white",
                        height=400, 
                        width=500,
                        font={"color": "darkblue", "family": "Arial"},
                        margin=dict(l=20, r=20, b=20, t=20, pad=10),
                        title_text='Distribution of income')
                        st.plotly_chart(fig_income)

                    with col2:
                        
                        # ============ Histogramme prix du bien ==============================================
                        good_price = df_info_pret['GOODS\' PRICE ($)'].loc[df_info_client['SK_ID_CURR']== client_id]
                        fig_good = px.histogram(df_info_pret, x="GOODS\' PRICE ($)" , hover_data=df_info_pret.columns)
                        fig_good.add_vline(x=int(good_price), line_dash = 'dash', line_color = 'firebrick')
                        fig_good.update_layout()
                        fig_good.update_layout(paper_bgcolor="white",
                        height=400, 
                        width=500,
                        font={"color": "darkblue", "family": "Arial"},
                        margin=dict(l=20, r=20, b=20, t=20, pad=10),
                        title_text='Distribution of age')
                        st.plotly_chart(fig_good)


                        # ============ Histogramme du revenu des clients ==============================================
                        credit_client = df_info_pret['AMOUNT REQUESTED ($)'].loc[df_info_client['SK_ID_CURR']== client_id]
                        fig_income = px.histogram(df_info_pret, x="AMOUNT REQUESTED ($)", hover_data=df_info_pret.columns)
                        fig_income.add_vline(x=int(credit_client), line_dash = 'dash', line_color = 'firebrick')
                        fig_income.update_layout()
                        fig_income.update_layout(paper_bgcolor="white",
                        height=400, 
                        width=500,
                        font={"color": "darkblue", "family": "Arial"},
                        margin=dict(l=20, r=20, b=20, t=20, pad=10),
                        title_text='Distribution of amount requested')
                        st.plotly_chart(fig_income)

st.sidebar.subheader('Clients similaires')
infos_clients_similaires()



# --------------------------------------------------------------------
# SHAP Values  
# --------------------------------------------------------------------

df_locator  = df_test_set[["SK_ID_CURR"]].copy().reset_index()
index = df_locator.loc[df_locator["SK_ID_CURR"] == client_id].index.to_list()[0]

def decision_explainer():
    ''' Affiche les valeurs SHAP globales et local
    '''
    html_shap="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #B8C8FA; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#B8C8FA; color:#0031CC;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Decision criteria
                  </h3>
            </div>
        </div>
        """
    titre = True
    
    # ====================== Facteurs d'influence de la décision basés sur SHAP =========================== 
    if st.sidebar.checkbox("See decision criteria ?"): 
        st.write(index)  
        if titre:
            st.markdown(html_shap, unsafe_allow_html=True)
            titre = False
            with st.spinner('**Printing comparing plots...**'): 
                col3, col4 = st.columns((1,1))
                with st.container():
                    st.markdown(""" 
                            * <span style="color:red">Red variables **disadvantage** credit score</span>.  
                            * <span style="color:blue"> Blue variables  **advantage** credit score</span>. 
                            """, unsafe_allow_html=True)

                    # Profil client
                    with col3:
                        col3.header('Client: ')

                        fig, ax = plt.subplots()
                        ax.set_title('Decision criteria for the client')
                        shap.plots.waterfall(shap_values[index])
                        col3.pyplot(fig, bbox_inches='tight')
                    with col4:
                        col4.header('Global: ')
                        fig, ax = plt.subplots()
                        ax.set_title('Decision criteria overall')
                        shap.plots.beeswarm(shap_values)
                        col4.pyplot(fig, bbox_inches='tight')

st.sidebar.subheader('Decision criteria')
decision_explainer()






