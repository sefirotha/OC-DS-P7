"""Application : Dashboard de Crédit Score

Auteur: Erwan Berthaud
Source: 
Local URL: 
Network URL: 
"""



import streamlit as st
from PIL import Image
import joblib
import pycaret
import requests
import pickle
from pycaret.classification import predict_model 
import plotly.graph_objects as go
import numpy as np
import datetime


# ====================================================================
# VARIABLES STATIQUES
# ====================================================================
# Répertoire de sauvegarde du meilleur modèle
best_model = "tuned_lgbm_f10.pkl"
# Test set 
file_test_set = "Data/Processed_data/test_df.pkl"
file_client_test = "Data/Processed_data/application_test.pkl"

# ====================================================================
# IMAGES
# ====================================================================
# Logo de l"entreprise
logo =  Image.open("Data/images/logo.png") 

# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header="""
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
st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
st.markdown("<style>body{background-color: #fbfff0}</style>",unsafe_allow_html=True)
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
            df_test_set.drop("TARGET", axis=1, inplace = True)

        with open(file_client_test, "rb") as df_client_test:
            df_client_test = pickle.load(df_client_test)
        
         
    return df_test_set, df_client_test

# Chargement des dataframes et du modèle
df_test_set, df_client_test = load() 

df_info_client = df_client_test[["SK_ID_CURR",
                                 "DAYS_BIRTH",
                                 "CODE_GENDER",
                                 "NAME_FAMILY_STATUS",
                                 "CNT_CHILDREN",
                                 "NAME_EDUCATION_TYPE", 
                                 "NAME_INCOME_TYPE",
                                 "DAYS_EMPLOYED",
                                 "AMT_INCOME_TOTAL",]].copy()

df_info_client["AGE"] = - np.round(df_info_client["DAYS_BIRTH"] / 365,0).astype(int)
df_info_client["YEARS EMPLOYED"] = - np.round(df_info_client["DAYS_EMPLOYED"] / 365,0).astype(int)
df_info_client = df_info_client[["SK_ID_CURR",
                                 "AGE",
                                 "CODE_GENDER",
                                 "NAME_FAMILY_STATUS",
                                 "CNT_CHILDREN",
                                 "NAME_EDUCATION_TYPE", 
                                 "NAME_INCOME_TYPE",
                                 "YEARS EMPLOYED",
                                 "AMT_INCOME_TOTAL",]]

df_info_client.rename(columns={"CODE_GENDER":"GENDER",
                        "NAME_FAMILY_STATUS":"FAMILY STATUS",
                        "CNT_CHILDREN":"NB OF CHILDREN",
                        "NAME_EDUCATION_TYPE": "EDUCATION", 
                        "NAME_INCOME_TYPE":"INCOME SOURCE",
                        "AMT_INCOME_TOTAL":"INCOME",
                        },
                        inplace = True)

df_info_pret = df_client_test[["SK_ID_CURR",
                                 "NAME_CONTRACT_TYPE",
                                 "AMT_CREDIT",
                                 "AMT_ANNUITY",
                                 "AMT_GOODS_PRICE",
                                 "NAME_HOUSING_TYPE",]].copy()

df_info_pret.rename(columns={"NAME_CONTRACT_TYPE":"CONTRACT TYPE",
                        "AMT_CREDIT":"AMOUNT REQUESTED ($)",
                        "AMT_ANNUITY":"ANNUITY ($)",
                        "AMT_GOODS_PRICE": "GOODS' PRICE ($)", 
                        "NAME_HOUSING_TYPE":"HOUSING TYPE",
                        },
                        inplace = True)

# ====================================================================
# CHOIX DU CLIENT
# ====================================================================

html_select_client="""
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
    col1, col2 = st.columns([1,3])
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
        st.table(client_info)
        # Infos principales sur la demande de prêt
        # st.write("*Demande de prêt*")
        client_pret = df_info_pret[df_info_pret["SK_ID_CURR"] == client_id].iloc[:, :]
        client_pret.set_index("SK_ID_CURR", inplace=True)
        st.table(client_pret)

# ====================================================================
# SCORE - PREDICTIONS
# ====================================================================

html_score="""
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

model_data = joblib.load(best_model)
# Sélection des variables du client étudié
X_test = df_test_set[df_test_set["SK_ID_CURR"] == client_id]
# Prédictions de probabiltés
y_proba = predict_model(model_data, data = X_test, raw_score = True)["prediction_score_1"]
# Score du client en pourcentage arrondi et nombre entier
#st.dataframe(y_proba)
score_client = int(np.rint(y_proba * 100))


# Graphique de jauge du cédit score ==========================================
fig_jauge = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    # Score du client en % df_dashboard["SCORE_CLIENT_%"]
    value = score_client,  
    domain = {"x": [0, 1], "y": [0, 1]},
    title = {"text": "Crédit score du client", "font": {"size": 24}},
    # Score des 10 voisins test set
    # delta = {"reference": score_moy_voisins_test,
    #          "increasing": {"color": "Crimson"},
    #          "decreasing": {"color": "Green"}},
    gauge = {"axis": {"range": [None, 100],
                      "tickwidth": 3,
                      "tickcolor": "darkblue"},
             "bar": {"color": "white", "thickness" : 0.25},
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
                           # Score du client en %
                           # df_dashboard["SCORE_CLIENT_%"]
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
            score_text = "Crédit score : EXCELLENT"
            st.success(score_text)
        elif 25 <= score_client < 50:
            score_text = "Crédit score : BON"
            st.success(score_text)
        elif 50 <= score_client < 75:
            score_text = "Crédit score : MOYEN"
            st.warning(score_text)
        else :
            score_text = "Crédit score : BAS"
            st.error(score_text)
        # st.write("")    
        # st.markdown(f"Crédit score moyen des 10 clients similaires : **{score_moy_voisins_test}**")
        # st.markdown(f"**{pourc_def_voisins_train}**% de clients voisins réellement défaillants dans l\"historique")
        # st.markdown(f"**{pourc_def_voisins_test}**% de clients voisins défaillants prédits pour les nouveaux clients")


dico_stats = {"Variable cible": "TARGET",
              "Type de prêt": "NAME_CONTRACT_TYPE",
              "Sexe": "CODE_GENDER",
              "Tél. professionnel": "FLAG_EMP_PHONE",
              "Note région où vit client": "REGION_RATING_CLIENT_W_CITY",
              "Niveau éducation du client": "NAME_EDUCATION_TYPE",
              "Profession du client": "OCCUPATION_TYPE",
              "Type d\"organisation de travail du client": "ORGANIZATION_TYPE",
              "Adresse du client = adresse de contact": "REG_CITY_NOT_LIVE_CITY",
              "Région du client = adresse professionnelle": "REG_CITY_NOT_WORK_CITY",
              "Adresse du client = adresse professionnelle": "LIVE_CITY_NOT_WORK_CITY",
              "Logement du client": "NAME_HOUSING_TYPE",
              "Statut familial": "NAME_FAMILY_STATUS",
              "Type de revenu du client": "NAME_INCOME_TYPE",
              "Client possède une maison ou appartement?": "FLAG_OWN_REALTY",
              "Accompagnateur lors de la demande de prêt?": "NAME_TYPE_SUITE",
              "Quel jour de la semaine le client a-t-il demandé le prêt ?": "WEEKDAY_APPR_PROCESS_START",
              "Le client a-t-il fourni un numéro de téléphone portable ?": "FLAG_MOBIL",
              "Le client a-t-il fourni un numéro de téléphone professionnel fixe ?": "FLAG_WORK_PHONE",
              "Le téléphone portable était-il joignable?": "FLAG_CONT_MOBILE",             
              "Le client a-t-il fourni un numéro de téléphone domicile fixe ?": "FLAG_PHONE",
              "Le client a-t-il fourni une adresse électronique": "FLAG_EMAIL",
              "Âge (ans)": "AGE_YEARS",
              "Combien d\"années avant la demande la personne a commencé son emploi actuel ?": "YEARS_EMPLOYED",
              "Combien de jours avant la demande le client a-t-il changé son enregistrement ?": "DAYS_REGISTRATION",
              "Combien de jours avant la demande le client a-t-il changé la pièce d\"identité avec laquelle il a demandé le prêt ?": "DAYS_ID_PUBLISH",
              "Prix du bien que le client a demandé": "AMT_GOODS_PRICE",
              "Nombre d\enfants?": "CNT_CHILDREN",
              "Revenu du client": "AMT_INCOME_TOTAL",
              "Montant du crédit du prêt": "AMT_CREDIT",
              "Annuité de prêt": "AMT_ANNUITY",
              "Âge de la voiture du client": "OWN_CAR_AGE",
              "Combien de membres de la famille a le client": "CNT_FAM_MEMBERS",
              "Population normalisée de la région où vit le client": "REGION_POPULATION_RELATIVE",
              "Notre évaluation de la région où vit le client (1 ou 2 ou 3)": "REGION_RATING_CLIENT",
              "Indicateur si l\"adresse permanente du client ne correspond pas à l\"adresse de contact": "REG_REGION_NOT_LIVE_REGION",
              "Indicateur si l\"adresse permanente du client ne correspond pas à l\"adresse professionnelle": "REG_REGION_NOT_WORK_REGION",
              "Indicateur si l\"adresse de contact du client ne correspond pas à l\"adresse de travail": "LIVE_REGION_NOT_WORK_REGION",
              "Combien de jours avant la demande le client a-t-il changé de téléphone ?": "DAYS_LAST_PHONE_CHANGE",
              "Statut des crédits déclarés par le Credit Bureau": "CREDIT_ACTIVE",
              "Devise recodée du crédit du Credit Bureau": "CREDIT_CURRENCY",
              "Type de crédit du Bureau de crédit (voiture ou argent liquide...)": "CREDIT_TYPE",
              "Combien d\années avant la demande actuelle le client a-t-il demandé un crédit au Credit Bureau ?": "YEARS_CREDIT",
              "Durée restante du crédit CB (en jours) au moment de la demande dans Crédit immobilier": "DAYS_CREDIT_ENDDATE",
              "Combien de jours avant la demande de prêt la dernière information sur la solvabilité du Credit Bureau a-t-elle été fournie ?": "DAYS_CREDIT_UPDATE",
              "Nombre de jours de retard sur le crédit CB au moment de la demande de prêt": "CREDIT_DAY_OVERDUE",
              "Montant maximal des impayés sur le crédit du Credit Bureau jusqu\"à présent": "AMT_CREDIT_MAX_OVERDUE",
              "Combien de fois le crédit du Bureau de crédit a-t-il été prolongé ?": "CNT_CREDIT_PROLONG",
              "Montant actuel du crédit du Credit Bureau": "AMT_CREDIT_SUM",
              "Dette actuelle sur le crédit du Credit Bureau": "AMT_CREDIT_SUM_DEBT",
              "Limite de crédit actuelle de la carte de crédit déclarée dans le Bureau de crédit": "AMT_CREDIT_SUM_LIMIT",
              "Montant actuel en retard sur le crédit du Bureau de crédit": "AMT_CREDIT_SUM_OVERDUE",
              "Annuité du crédit du Credit Bureau": "AMT_ANNUITY",
              "Statut du prêt du Credit Bureau durant le mois": "STATUS",
              "Mois du solde par rapport à la date de la demande": "MONTHS_BALANCE",
              "Statut du contrat au cours du mois": "NAME_CONTRACT_STATUS",
              "Solde au cours du mois du crédit précédent": "AMT_BALANCE",
              "Montant total à recevoir sur le crédit précédent": "AMT_TOTAL_RECEIVABLE",
              "Nombre d\"échéances payées sur le crédit précédent": "CNT_INSTALMENT_MATURE_CUM",
              "Mois du solde par rapport à la date d\"application": "MONTHS_BALANCE",
              "Limite de la carte de crédit au cours du mois du crédit précédent": "AMT_CREDIT_LIMIT_ACTUAL",
              "Montant retiré au guichet automatique pendant le mois du crédit précédent": "AMT_DRAWINGS_ATM_CURRENT",
              "Montant prélevé au cours du mois du crédit précédent": "AMT_DRAWINGS_CURRENT",
              "Montant des autres prélèvements au cours du mois du crédit précédent": "AMT_DRAWINGS_OTHER_CURRENT",
              "Montant des prélèvements ou des achats de marchandises au cours du mois de la crédibilité précédente": "AMT_DRAWINGS_POS_CURRENT",
              "Versement minimal pour ce mois du crédit précédent": "AMT_INST_MIN_REGULARITY",
              "Combien le client a-t-il payé pendant le mois sur le crédit précédent ?": "AMT_PAYMENT_CURRENT",
              "Combien le client a-t-il payé au total pendant le mois sur le crédit précédent ?": "AMT_PAYMENT_TOTAL_CURRENT",
              "Montant à recevoir pour le principal du crédit précédent": "AMT_RECEIVABLE_PRINCIPAL",
              "Montant à recevoir sur le crédit précédent": "AMT_RECIVABLE", 
              "Nombre de retraits au guichet automatique durant ce mois sur le crédit précédent": "CNT_DRAWINGS_ATM_CURRENT",
              "Nombre de retraits pendant ce mois sur le crédit précédent": "CNT_DRAWINGS_CURRENT",
              "Nombre d\"autres retraits au cours de ce mois sur le crédit précédent": "CNT_DRAWINGS_OTHER_CURRENT",
              "Nombre de retraits de marchandises durant ce mois sur le crédit précédent": "CNT_DRAWINGS_POS_CURRENT",
              "DPD (jours de retard) au cours du mois sur le crédit précédent": "SK_DPD",
              "DPD (Days past due) au cours du mois avec tolérance (les dettes avec de faibles montants de prêt sont ignorées) du crédit précédent": "SK_DPD_DEF",
              "La date à laquelle le versement du crédit précédent était censé être payé (par rapport à la date de demande du prêt actuel)": "DAYS_INSTALMENT",
              "Quand les échéances du crédit précédent ont-elles été effectivement payées (par rapport à la date de demande du prêt actuel) ?": "DAYS_ENTRY_PAYMENT",
              "Version du calendrier des versements (0 pour la carte de crédit) du crédit précédent": "NUM_INSTALMENT_VERSION",
              "Sur quel versement nous observons le paiement": "NUM_INSTALMENT_NUMBER",
              "Quel était le montant de l\"acompte prescrit du crédit précédent sur cet acompte ?": "AMT_INSTALMENT",
              "Ce que le client a effectivement payé sur le crédit précédent pour ce versement": "AMT_PAYMENT",
              "Statut du contrat au cours du mois": "NAME_CONTRACT_STATUS",
              "Durée du crédit précédent (peut changer avec le temps)": "CNT_INSTALMENT",
              "Versements restant à payer sur le crédit précédent": "CNT_INSTALMENT_FUTURE",
              "EXT_SOURCE_1": "EXT_SOURCE_1",
              "EXT_SOURCE_2": "EXT_SOURCE_2",
              "EXT_SOURCE_3": "EXT_SOURCE_3",
              "FLAG_DOCUMENT_2": "FLAG_DOCUMENT_2",
              "FLAG_DOCUMENT_3": "FLAG_DOCUMENT_3",
              "FLAG_DOCUMENT_4": "FLAG_DOCUMENT_4",
              "FLAG_DOCUMENT_5": "FLAG_DOCUMENT_5",
              "FLAG_DOCUMENT_6": "FLAG_DOCUMENT_6",
              "FLAG_DOCUMENT_7": "FLAG_DOCUMENT_7",
              "FLAG_DOCUMENT_8": "FLAG_DOCUMENT_8",
              "FLAG_DOCUMENT_9": "FLAG_DOCUMENT_9",
              "FLAG_DOCUMENT_10": "FLAG_DOCUMENT_10",
              "FLAG_DOCUMENT_11": "FLAG_DOCUMENT_11",
              "FLAG_DOCUMENT_12": "FLAG_DOCUMENT_12",
              "FLAG_DOCUMENT_13": "FLAG_DOCUMENT_13",
              "FLAG_DOCUMENT_14": "FLAG_DOCUMENT_14",
              "FLAG_DOCUMENT_15": "FLAG_DOCUMENT_15",
              "FLAG_DOCUMENT_16": "FLAG_DOCUMENT_16",
              "FLAG_DOCUMENT_17": "FLAG_DOCUMENT_17",
              "FLAG_DOCUMENT_18": "FLAG_DOCUMENT_18",
              "FLAG_DOCUMENT_19": "FLAG_DOCUMENT_19",
              "FLAG_DOCUMENT_20": "FLAG_DOCUMENT_20",
              "FLAG_DOCUMENT_21": "FLAG_DOCUMENT_21",
              "FONDKAPREMONT_MODE": "FONDKAPREMONT_MODE",
              "HOUSETYPE_MODE": "HOUSETYPE_MODE",
              "WALLSMATERIAL_MODE": "WALLSMATERIAL_MODE",
              "EMERGENCYSTATE_MODE": "EMERGENCYSTATE_MODE",
              "FLOORSMAX_AVG": "FLOORSMAX_AVG",
              "FLOORSMAX_MEDI": "FLOORSMAX_MEDI",
              "FLOORSMAX_MODE": "FLOORSMAX_MODE",
              "FLOORSMIN_MODE": "FLOORSMIN_MODE",
              "FLOORSMIN_AVG": "FLOORSMIN_AVG",
              "FLOORSMIN_MEDI": "FLOORSMIN_MEDI",
              "APARTMENTS_AVG": "APARTMENTS_AVG",
              "APARTMENTS_MEDI": "APARTMENTS_MEDI",
              "APARTMENTS_MODE": "APARTMENTS_MODE",
              "BASEMENTAREA_AVG": "BASEMENTAREA_AVG",
              "BASEMENTAREA_MEDI": "BASEMENTAREA_MEDI",
              "BASEMENTAREA_MODE": "BASEMENTAREA_MODE",
              "YEARS_BEGINEXPLUATATION_AVG": "YEARS_BEGINEXPLUATATION_AVG",
              "YEARS_BEGINEXPLUATATION_MODE": "YEARS_BEGINEXPLUATATION_MODE",
              "YEARS_BEGINEXPLUATATION_MEDI": "YEARS_BEGINEXPLUATATION_MEDI",
              "YEARS_BUILD_AVG": "YEARS_BUILD_AVG",
              "YEARS_BUILD_MODE": "YEARS_BUILD_MODE",
              "YEARS_BUILD_MEDI": "YEARS_BUILD_MEDI",
              "COMMONAREA_AVG": "COMMONAREA_AVG",
              "COMMONAREA_MEDI": "COMMONAREA_MEDI",
              "COMMONAREA_MODE": "COMMONAREA_MODE",
              "ELEVATORS_AVG": "ELEVATORS_AVG",
              "ELEVATORS_MODE": "ELEVATORS_MODE",
              "ELEVATORS_MEDI": "ELEVATORS_MEDI",
              "ENTRANCES_AVG": "ENTRANCES_AVG",
              "ENTRANCES_MODE": "ENTRANCES_MODE",
              "ENTRANCES_MEDI": "ENTRANCES_MEDI",
              "LANDAREA_AVG": "LANDAREA_AVG",
              "LANDAREA_MEDI": "LANDAREA_MEDI",
              "LANDAREA_MODE": "LANDAREA_MODE",
              "LIVINGAPARTMENTS_AVG": "LIVINGAPARTMENTS_AVG",
              "LIVINGAPARTMENTS_MODE": "LIVINGAPARTMENTS_MODE",
              "LIVINGAPARTMENTS_MEDI": "LIVINGAPARTMENTS_MEDI",
              "LIVINGAREA_AVG": "LIVINGAREA_AVG",
              "LIVINGAREA_MODE": "LIVINGAREA_MODE",
              "NONLIVINGAPARTMENTS_AVG": "NONLIVINGAPARTMENTS_AVG",
              "NONLIVINGAPARTMENTS_MODE": "NONLIVINGAPARTMENTS_MODE",
              "NONLIVINGAREA_AVG": "NONLIVINGAREA_AVG",
              "NONLIVINGAREA_MEDI": "NONLIVINGAREA_MEDI",
              "TOTALAREA_MODE": "TOTALAREA_MODE",
              "OBS_30_CNT_SOCIAL_CIRCLE": "OBS_30_CNT_SOCIAL_CIRCLE",
              "DEF_30_CNT_SOCIAL_CIRCLE": "DEF_30_CNT_SOCIAL_CIRCLE",
              "DEF_60_CNT_SOCIAL_CIRCLE": "DEF_60_CNT_SOCIAL_CIRCLE",
              "AMT_REQ_CREDIT_BUREAU_HOUR": "AMT_REQ_CREDIT_BUREAU_HOUR",
              "AMT_REQ_CREDIT_BUREAU_DAY": "AMT_REQ_CREDIT_BUREAU_DAY",
              "AMT_REQ_CREDIT_BUREAU_WEEK": "AMT_REQ_CREDIT_BUREAU_WEEK",
              "AMT_REQ_CREDIT_BUREAU_MON": "AMT_REQ_CREDIT_BUREAU_MON",
              "AMT_REQ_CREDIT_BUREAU_QRT": "AMT_REQ_CREDIT_BUREAU_QRT",
              "AMT_REQ_CREDIT_BUREAU_YEAR": "AMT_REQ_CREDIT_BUREAU_YEAR",
              "DAYS_ENDDATE_FACT": "DAYS_ENDDATE_FACT"}