import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

from feature_engine.encoding import CountFrequencyEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_auc_score, log_loss

# ==============================
# Funções auxiliares
# ==============================

def detectar_outliers(df: pd.DataFrame, metodo='zscore', threshold=3) -> dict:
    variaveis_com_outliers = {}
    for col in df.select_dtypes(include=["float", "int"]).columns:
        serie = df[col].dropna()
        if serie.nunique() > 2:
            if metodo == "zscore":
                media = serie.mean()
                desvio = serie.std()
                z_scores = (serie - media) / desvio
                outliers = serie[abs(z_scores) > threshold]
            else:
                raise ValueError("Método inválido. Use 'zscore'.")
            if len(outliers) > 0:
                variaveis_com_outliers[col] = len(outliers)
    return variaveis_com_outliers

def capping(df: pd.DataFrame, variaveis_outliers: dict, metodo="zscore", threshold=3) -> pd.DataFrame:
    df_capped = df.copy()
    for col in variaveis_outliers.keys():
        serie = df_capped[col]
        if metodo == "zscore":
            media = serie.mean()
            desvio = serie.std()
            limite_inferior = media - threshold * desvio
            limite_superior = media + threshold * desvio
        else:
            raise ValueError("Método inválido. Use 'zscore'.")
        # Aplica capping
        df_capped[col] = serie.clip(limite_inferior, limite_superior)
    return df_capped

def preprocess_data(df):
    df['log_Transaction_Amount'] = np.log1p(df['Transaction_Amount'])

    # Detectar outliers e remover linhas que ultrapassam threshold
    outliers = detectar_outliers(df[['log_Transaction_Amount']], metodo='zscore', threshold=3)

    # Aplicar capping nas demais variáveis numéricas (opcional)
    df = capping(df, outliers, metodo='zscore', threshold=3)

    # Conversão de Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['User_ID', 'Timestamp'])

    # Agregações por usuário
    agg_funcs = {'Failed_Transaction_Count_7d': ['max']}
    user_agg = df.groupby('User_ID').agg(agg_funcs)
    user_agg.columns = ['_'.join(col) for col in user_agg.columns]
    user_agg.reset_index(inplace=True)
    df = df.merge(user_agg, on='User_ID', how='left')

    # Features de taxa
    df['Failed_Transaction_Rate'] = df['Failed_Transaction_Count_7d'] / (df['Daily_Transaction_Count'] + 1)

    # Encoding de variáveis categóricas
    var_cat = df.select_dtypes(include=['object']).drop(columns=['Transaction_ID', 'User_ID']).columns.to_list()
    encoder = CountFrequencyEncoder(encoding_method='frequency', variables=var_cat, ignore_format=True)
    df_encoded = encoder.fit_transform(df)

    # Seleção de variáveis finais
    selected_features = [
        'Failed_Transaction_Rate',
        'Transaction_Type',
        'Device_Type',
        'Authentication_Method',
        'Merchant_Category',
        'Failed_Transaction_Count_7d_max',
        'Card_Type',
        'Fraud_Label'
    ]

    return df_encoded[selected_features]


def add_download_button(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download com Risco de Fraude",
        data=csv,
        file_name="fraud_risk_predictions.csv",
        mime="text/csv",
    )
    
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")  # garante caminho correto
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# ==============================
# Streamlit App
# ==============================

st.title("Classificação de Fraudes em Transações Financeiras")

uploaded_file = st.file_uploader("Envie um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Prévia dos Dados Originais")
    st.dataframe(df.head())

    # Pré-processamento
    df_processed = preprocess_data(df)

    st.subheader("✅ Dados Após Pré-processamento")
    st.dataframe(df_processed.head())

    # Separar features e target (assumindo que o target já exista no dataframe original)
    if "Fraud_Label" in df.columns:
        X = df_processed.drop(columns=['Fraud_Label'])
        y = df_processed["Fraud_Label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, random_state=23
        )

        # Carregar modelo salvo
        model = load_model()

        # Predições no teste
        y_probs = model.predict_proba(X_test)[:, 1]
    
        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
            
        y_pred = (y_probs >= best_threshold).astype(int)
        
        #y_pred = model.predict(X_test)

        # Métricas
        st.subheader("📈 Métricas do Modelo")
        st.text(classification_report(y_test, y_pred))
        st.write(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
        st.write(f"Log Loss: {log_loss(y_test, y_probs):.4f}")
        
        # Matriz de Confusão
        st.subheader("📊 Matriz de Confusão")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        st.pyplot(fig)

        # Adicionar coluna de risco de fraude no DataFrame original
        df["Fraud_Risk"] = model.predict_proba(df_processed.drop(columns=['Fraud_Label']))[:, 1]
        df = df.drop(columns=['Risk_Score'])
        # Botão de download
        add_download_button(df)

    else:
        st.warning("⚠️ O dataset precisa conter a coluna `Fraud_Label` como variável alvo.")
