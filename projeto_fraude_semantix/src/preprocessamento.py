import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

def detectar_outliers(df: pd.DataFrame, metodo='IQR', threshold=1.5) -> dict:
    '''
    Detecta variáveis com outliers em um DataFrame.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados numéricos.
    metodo : str, default="IQR"
        Método para detectar outliers ("IQR" ou "zscore").
    threshold : float
        Limite para identificar outliers.
        - Para IQR, o padrão é 1.5.
        - Para zscore, geralmente usa-se 3.

    Retorna:
    --------
    dict
        Dicionário com o nome da variável e a contagem de outliers.
    '''
    variaveis_com_outliers = {}

    for col in df.select_dtypes(include=["float", "int"]).columns:
        serie = df[col].dropna()
        
        if serie.nunique() > 2:

            if metodo == "IQR":
                Q1 = serie.quantile(0.25)
                Q3 = serie.quantile(0.75)
                IQR = Q3 - Q1
                limite_inferior = Q1 - threshold * IQR
                limite_superior = Q3 + threshold * IQR
                outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]

            elif metodo == "zscore":
                media = serie.mean()
                desvio = serie.std()
                z_scores = (serie - media) / desvio
                outliers = serie[abs(z_scores) > threshold]

            else:
                raise ValueError("Método inválido. Use 'IQR' ou 'zscore'.")

            if len(outliers) > 0:
                variaveis_com_outliers[col] = len(outliers)
        
        else: 
            continue

    return variaveis_com_outliers

def capping(df: pd.DataFrame, variaveis_outliers: dict, metodo="IQR", threshold=1.5) -> pd.DataFrame:
    '''
    Aplica capping (winsorização) nas variáveis que possuem outliers.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados.
    variaveis_outliers : dict
        Dicionário retornado pela função detectar_outliers (variável -> nº de outliers).
    metodo : str, default="IQR"
        Método para calcular limites ("IQR" ou "zscore").
    threshold : float
        Limite para definir outliers.

    Retorna:
    --------
    df_capped : pandas.DataFrame
        DataFrame com capping aplicado.
    '''
    df_capped = df.copy()

    for col in variaveis_outliers.keys():
        serie = df_capped[col]

        if metodo == "IQR":
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - threshold * IQR
            limite_superior = Q3 + threshold * IQR

        elif metodo == "zscore":
            media = serie.mean()
            desvio = serie.std()
            limite_inferior = media - threshold * desvio
            limite_superior = media + threshold * desvio

        else:
            raise ValueError("Método inválido. Use 'IQR' ou 'zscore'.")

        # Aplica capping
        df_capped[col] = serie.clip(limite_inferior, limite_superior)

    return df_capped
