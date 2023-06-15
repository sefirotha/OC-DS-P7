# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("model")

# Create input/output pydantic models
input_model = create_model("api_input", **{'CODE_GENDER': 0.0, 'FLAG_OWN_CAR': 1.0, 'FLAG_OWN_REALTY': 0.0, 'CNT_CHILDREN': 0.0, 'AMT_CREDIT': 267102.0, 'AMT_ANNUITY': 21415.5, 'AMT_GOODS_PRICE': 247500.0, 'REGION_POPULATION_RELATIVE': 0.01522064208984375, 'DAYS_BIRTH': -11200.0, 'DAYS_EMPLOYED': -674.0, 'DAYS_REGISTRATION': -595.0, 'DAYS_ID_PUBLISH': -3708.0, 'FLAG_EMP_PHONE': 1.0, 'FLAG_WORK_PHONE': 1.0, 'FLAG_PHONE': 1.0, 'CNT_FAM_MEMBERS': 2.0, 'REGION_RATING_CLIENT': 2.0, 'REGION_RATING_CLIENT_W_CITY': 2.0, 'REG_CITY_NOT_LIVE_CITY': 0.0, 'REG_CITY_NOT_WORK_CITY': 1.0, 'EXT_SOURCE_2': 0.53076171875, 'EXT_SOURCE_3': 0.59375, 'FLOORSMAX_MODE': 0.0029125213623046875, 'OBS_30_CNT_SOCIAL_CIRCLE': 0.0, 'DEF_30_CNT_SOCIAL_CIRCLE': 0.0, 'OBS_60_CNT_SOCIAL_CIRCLE': 0.0, 'DEF_60_CNT_SOCIAL_CIRCLE': 0.0, 'DAYS_LAST_PHONE_CHANGE': -771.0, 'FLAG_DOCUMENT_3': 1.0, 'AMT_REQ_CREDIT_BUREAU_QRT': 1.0, 'AMT_REQ_CREDIT_BUREAU_YEAR': 2.0, 'NAME_CONTRACT_TYPE_Cashloans': 1.0, 'NAME_TYPE_SUITE_Unaccompanied': 1.0, 'NAME_INCOME_TYPE_Commercialassociate': 0.0, 'NAME_INCOME_TYPE_Pensioner': 0.0, 'NAME_INCOME_TYPE_Working': 1.0, 'NAME_EDUCATION_TYPE_Highereducation': 0.0, 'NAME_EDUCATION_TYPE_Secondarysecondaryspecial': 1.0, 'NAME_FAMILY_STATUS_Married': 1.0, 'NAME_FAMILY_STATUS_Singlenotmarried': 0.0, 'NAME_HOUSING_TYPE_Houseapartment': 1.0, 'OCCUPATION_TYPE_Corestaff': 0.0, 'OCCUPATION_TYPE_Drivers': 1.0, 'OCCUPATION_TYPE_Laborers': 0.0, 'OCCUPATION_TYPE_Salesstaff': 0.0, 'WEEKDAY_APPR_PROCESS_START_FRIDAY': 0.0, 'WEEKDAY_APPR_PROCESS_START_MONDAY': 1.0, 'WEEKDAY_APPR_PROCESS_START_TUESDAY': 0.0, 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY': 0.0, 'ORGANIZATION_TYPE_BusinessEntityType3': 0.0, 'ORGANIZATION_TYPE_Selfemployed': 0.0, 'ORGANIZATION_TYPE_XNA': 0.0, 'HOUSETYPE_MODE_blockofflats': 0.0, 'WALLSMATERIAL_MODE_Panel': 0.0, 'WALLSMATERIAL_MODE_Stonebrick': 0.0, 'EMERGENCYSTATE_MODE_No': 0.0, 'DAYS_EMPLOYED_PERC': 0.0601806640625, 'INCOME_CREDIT_PERC': 0.286376953125, 'ANNUITY_INCOME_PERC': 0.280029296875, 'PAYMENT_RATE': 0.0802001953125, 'BURO_DAYS_CREDIT_MAX': -1157.0, 'BURO_DAYS_CREDIT_MEAN': -1157.0, 'BURO_DAYS_CREDIT_ENDDATE_MAX': -159.0, 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN': 0.0, 'BURO_AMT_CREDIT_SUM_SUM': 407021.65625, 'BURO_AMT_CREDIT_SUM_DEBT_MEAN': 0.0, 'BURO_AMT_CREDIT_SUM_DEBT_SUM': 0.0, 'BURO_AMT_CREDIT_SUM_OVERDUE_MEAN': 0.0, 'BURO_AMT_CREDIT_SUM_LIMIT_MEAN': 0.0, 'BURO_AMT_CREDIT_SUM_LIMIT_SUM': 0.0, 'BURO_CREDIT_ACTIVE_Active_MEAN': 0.66650390625, 'BURO_CREDIT_ACTIVE_Closed_MEAN': 0.333251953125, 'BURO_CREDIT_TYPE_Cashloannonearmarked_MEAN': 0.0, 'BURO_CREDIT_TYPE_Microloan_MEAN': 0.0, 'BURO_CREDIT_TYPE_Mortgage_MEAN': 0.0, 'ACTIVE_DAYS_CREDIT_MAX': -1157.0, 'ACTIVE_DAYS_CREDIT_MEAN': -1157.0, 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN': -463.0, 'ACTIVE_DAYS_CREDIT_ENDDATE_MAX': -159.0, 'ACTIVE_DAYS_CREDIT_ENDDATE_MEAN': -311.0, 'ACTIVE_AMT_CREDIT_SUM_SUM': 337500.0, 'ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN': 0.0, 'ACTIVE_AMT_CREDIT_SUM_DEBT_SUM': 0.0, 'ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN': 0.0, 'ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN': 0.0, 'ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM': 0.0, 'CLOSED_DAYS_CREDIT_MAX': -1157.0, 'CLOSED_AMT_CREDIT_SUM_MEAN': 69521.671875, 'CLOSED_AMT_CREDIT_SUM_SUM': 69521.671875, 'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN': 0.0, 'CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN': 0.0, 'CLOSED_MONTHS_BALANCE_SIZE_SUM': 0.0, 'PREV_AMT_ANNUITY_MIN': 2987.550048828125, 'PREV_AMT_APPLICATION_MIN': 0.0, 'PREV_AMT_CREDIT_MIN': 0.0, 'PREV_APP_CREDIT_PERC_MIN': 0.81689453125, 'PREV_APP_CREDIT_PERC_MAX': 1.0, 'PREV_APP_CREDIT_PERC_MEAN': 0.8974609375, 'PREV_APP_CREDIT_PERC_VAR': 0.00872802734375, 'PREV_AMT_DOWN_PAYMENT_MAX': 2110.5, 'PREV_RATE_DOWN_PAYMENT_MIN': 0.08782958984375, 'PREV_RATE_DOWN_PAYMENT_MAX': 0.08782958984375, 'PREV_DAYS_DECISION_MIN': -771.0, 'PREV_CNT_PAYMENT_MEAN': 8.0, 'PREV_NAME_CONTRACT_TYPE_Revolvingloans_MEAN': 0.25, 'PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN': 0.0, 'PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN': 0.0, 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN': 0.25, 'PREV_CODE_REJECT_REASON_XAP_MEAN': 0.75, 'PREV_NAME_CLIENT_TYPE_New_MEAN': 0.25, 'PREV_NAME_CLIENT_TYPE_Repeater_MEAN': 0.75, 'PREV_NAME_GOODS_CATEGORY_Mobile_MEAN': 0.0, 'PREV_NAME_PORTFOLIO_Cards_MEAN': 0.25, 'PREV_NAME_PORTFOLIO_XNA_MEAN': 0.25, 'PREV_NAME_PRODUCT_TYPE_walkin_MEAN': 0.25, 'PREV_CHANNEL_TYPE_APCashloan_MEAN': 0.5, 'PREV_CHANNEL_TYPE_Contactcenter_MEAN': 0.0, 'PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN': 0.5, 'PREV_NAME_YIELD_GROUP_high_MEAN': 0.5, 'PREV_NAME_YIELD_GROUP_low_action_MEAN': 0.0, 'PREV_PRODUCT_COMBINATION_CardStreet_MEAN': 0.0, 'PREV_PRODUCT_COMBINATION_CardXSell_MEAN': 0.25, 'PREV_PRODUCT_COMBINATION_Cash_MEAN': 0.25, 'PREV_PRODUCT_COMBINATION_CashXSellhigh_MEAN': 0.0, 'PREV_PRODUCT_COMBINATION_CashXSelllow_MEAN': 0.0, 'PREV_PRODUCT_COMBINATION_POSindustrywithinterest_MEAN': 0.0, 'APPROVED_AMT_ANNUITY_MAX': 6536.33984375, 'APPROVED_AMT_ANNUITY_MEAN': 4674.6298828125, 'APPROVED_APP_CREDIT_PERC_MIN': 0.81689453125, 'APPROVED_APP_CREDIT_PERC_MAX': 1.0, 'APPROVED_AMT_DOWN_PAYMENT_MIN': 2110.5, 'APPROVED_AMT_DOWN_PAYMENT_MAX': 2110.5, 'APPROVED_HOUR_APPR_PROCESS_START_MAX': 11.0, 'APPROVED_RATE_DOWN_PAYMENT_MIN': 0.08782958984375, 'APPROVED_RATE_DOWN_PAYMENT_MAX': 0.08782958984375, 'APPROVED_DAYS_DECISION_MIN': -771.0, 'APPROVED_CNT_PAYMENT_MEAN': 8.0, 'POS_MONTHS_BALANCE_MAX': -7.0, 'POS_MONTHS_BALANCE_MEAN': -16.375, 'POS_MONTHS_BALANCE_SIZE': 19.0, 'POS_SK_DPD_MAX': 0.0, 'POS_SK_DPD_DEF_MAX': 0.0, 'POS_SK_DPD_DEF_MEAN': 0.0, 'POS_NAME_CONTRACT_STATUS_Active_MEAN': 0.89453125, 'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE': 3.0, 'INSTAL_DPD_MAX': 0.0, 'INSTAL_DPD_MEAN': 0.0, 'INSTAL_DPD_SUM': 0.0, 'INSTAL_DBD_SUM': 209.0, 'INSTAL_PAYMENT_PERC_MEAN': 1.0, 'INSTAL_PAYMENT_DIFF_MEAN': 0.0, 'INSTAL_PAYMENT_DIFF_SUM': 0.0, 'INSTAL_AMT_INSTALMENT_SUM': 138222.578125, 'INSTAL_AMT_PAYMENT_MIN': 97.42500305175781, 'INSTAL_AMT_PAYMENT_MEAN': 4936.52099609375, 'INSTAL_AMT_PAYMENT_SUM': 138222.578125, 'INSTAL_DAYS_ENTRY_PAYMENT_MAX': -26.0, 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN': -305.5, 'INSTAL_DAYS_ENTRY_PAYMENT_SUM': -8554.0})
output_model = create_model("api_output", prediction=1.0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    prediction = predict_model(model, data=data, raw_score = True)
    return {"prediction": prediction["prediction_label"].iloc[0]}


# Define predict_proba function
@app.post("/predict_proba", response_model=output_model)
def predict_proba(data: input_model):
    data = pd.DataFrame([data.dict()])
    prediction_proba = predict_model(model, data=data, raw_score = True)
    return {"prediction proba": prediction_proba["prediction_score_1"].iloc[0]}

# for test only

# Attention au host et au port
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
