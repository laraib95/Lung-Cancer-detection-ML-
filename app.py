import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import base64

# Load the saved model and selected features
with open("C:/Users/personal/Desktop/Lung Cancer FYP/final_gaussian_nb_model.pkl", "rb") as file:
    model_data = pickle.load(file)
    lung_cancer_model = model_data['model']
    selected_features = model_data['selected_features']

st.set_page_config(page_title="Lung Cancer Detection", layout="wide")

# Function to set background image
def set_bg_hack(main_bg):
    main_bg_ext = "jpeg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image
set_bg_hack("C://Users//personal//Desktop//Lung Cancer FYP//Background.jpeg")

# Rest of your Streamlit app code
st.title("Lung Cancer Detection Web App")
st.sidebar.title("Predicting Lung Cancer using ML models")
with st.sidebar:
    selected = option_menu('Lung Cancer Prediction System',
                           ['Lung Cancer Detection',
                            'About Lung Cancer Detection Project',
                            'Binary Classification Categories',
                            'About Us'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'lungs-fill', 'person'],
                           default_index=0)

# Function to get user input
def get_user_input():
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        smoking = st.selectbox("Smoking", ["yes", "no"])
        yellow_fingers = st.selectbox("Yellow Fingers", ["yes", "no"])
        anxiety = st.selectbox("Anxiety", ["yes", "no"])
    with col2:
        peer_pressure = st.selectbox("Peer Pressure", ["yes", "no"])
        chronic_disease = st.selectbox("Chronic Disease", ["yes", "no"])
        fatigue = st.selectbox("Fatigue", ["yes", "no"])
        allergy = st.selectbox("Allergy", ["yes", "no"])
        wheezing = st.selectbox("Wheezing", ["yes", "no"])
    with col3:
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["yes", "no"])
        coughing = st.selectbox("Coughing", ["yes", "no"])
        shortness_of_breath = st.selectbox("Shortness of Breath", ["yes", "no"])
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["yes", "no"])
        chest_pain = st.selectbox("Chest Pain", ["yes", "no"])
    features_list = [
        gender, smoking, yellow_fingers, anxiety, peer_pressure,
        chronic_disease, fatigue, allergy, wheezing, alcohol_consumption,
        coughing, shortness_of_breath, swallowing_difficulty, chest_pain
    ]
    return np.array(features_list, dtype=object)

# Function to predict cancer risk
def predict_cancer_risk(features):
    if features is not None:
        features[0] = 1 if features[0] == "M" else 0
        for i in range(1, len(features)):
            features[i] = 1 if features[i] == "yes" else 0

        # Select only the features that were used for training
        features = features[selected_features]
        features = features.reshape(1, -1)

        cancer_pred = ''
        if st.button('Lung Cancer Prediction Result'):
            cancer_pred = lung_cancer_model.predict(features)
            if cancer_pred[0] == 1:
                cancer_diagnosis = "Lung cancer is not diagnosed."
            else:
                cancer_diagnosis = "Person is diagnosed with Lung cancer."

            st.success(f"The prediction is: {cancer_diagnosis}")

if selected == "Lung Cancer Detection":
    features = get_user_input()
    if features.any():
        predict_cancer_risk(features)
elif selected == "About Lung Cancer Detection Project":
    st.header("Introduction of Lung Cancer Detection Project")
    st.subheader("About Lung Cancer Model")
    st.write("""Lung Cancer using GA-NB leverages advanced machine learning technique to tackle the
             critical challenge of early lung cancer detection. This innovative model combines the power 
             of Gneteic Algorithm for hyperparameter optimization with the accuracy of Naive Bayes 
             for Classification. By analyzing diverse medical data sources including electronic health 
             records and medical reports, the model excels at identifying potential malignancies in lung 
             scans.Its ability to extract subtle patterns from unstructured text data makes it a promising 
             toolfor healthcare prefessionals, contributing to early interventions and improved patient
             outcomes.""")
    st.subheader("Methodology")
    st.write("""By collecting diverse datasets of medical reports and records. Then we pre-process this 
             textual data to extract rel;evant features. Next, we employ a hybrid approach, combining 
             Genetic Algorithm(GC) and Naive Bayes(NB), to optimize the NAive Bayes model's hyperparameters
              effectively. This optimized model is trained on the dataset to classify lung scans as 
             cancerous or non-cancerous.We evaluate the model's performance using various metrices 
             like accuracy and recall, ensuring its effectiveness in early and accurate lung cancer detection.""")
elif selected == "Binary Classification Categories":
    st.header("Causes of Lung cancer :")
    st.write("""Categaries of aur Dataset are :""")
    st.write("""Smoking : smoking causes lung cancer by introducing harmful chemicals into the lungs,
                which can damage cells and lead to the development of  tumors.""")
    st.write("""Yellow_fingers : fingers may turn yellow in lung cancer due to poor circulation or as a side
                effect of acvanced disease. but itis not a common symptom.  """)
    st.write("""Chronic_Disease : Chronic diseases like COPD or emphysema, can damage lung tissue over 
                time, increasing  the risk of lung cancer development. """)
    st.write("""Fatigue: Lung cancer can cause fatigue due to factors like tumor growth, anemia, and the body's
                energy redirected to fighting the disease. """)
    st.write("""Allergy: Lung cancer itself doesn't cause allergies directly, but weakened immune systems due to
                cancer or treatments may make individuals more susceptible to allergies. """)
    st.write("""Wheezing: Lung cancer can obstruct air passages, leading to wheezing as air struggles to pass through
                narrowed or blocked respiratory pathways. """)
    st.write("""Alcohol_consuming: Excessive alcohol consumption can weaken the immune system and lead to
                harmful substances inhaled while drinking, potentially increasing lung cancer risk. """)
    st.write("""Coughing: Lung cancer can irritate the airways, leading to chronic coughing as the body attempts to
                clear mucus and foreign particles from the lungs """)
    st.write("""Shortness of Breath: This can occur if lung cancer blocks or narrows an airway, or if fluid from a 
             lung tumor builds up in the chest. """)
    st.write(""" Swallowing Difficulty: Lung cancer that affects the esophagus can lead to difficulty swallowing.""")
    st.write("""Chest Pain: Lung cancer may cause chest pain if it spreads to the chest wall, ribs, or nearby nerves 
             and muscles. """)
    st.write("""Peer Pressure: Though not directly linked to lung cancer, peer pressure can influence behaviors such
              as smoking, which is a significant risk factor for developing lung cancer. """)
    st.write("""Gender: While lung cancer affects both genders, historically, it has been more prevalent among males.
              However, the gap in lung cancer rates between genders is narrowing. """)
elif selected == "About Us":
    st.header("About Us")
    st.write("""            Meet the team           """)
    st.subheader("Laraib Masood : Fa20/BSCS/534/Section C")
    st.subheader("Amina Nawaz   : Fa20/BSCS/093/Section C")
    st.write("""          contact us           """)
    st.subheader("Laraib Masood :   Fa20-bscs-534@lgu.edu.pk")
    st.subheader("Amina Nawaz   :   Fa20-bscs-093@lgu.edu.pk")
