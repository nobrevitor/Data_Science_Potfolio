import pandas as pd 
from sklearn.preprocessing import StandardScaler

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um arquivo CSV."""
    return pd.read_csv(caminho)

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove valores ausentes e possíveis duplicatas."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def normalizar_variaveis(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    """Normaliza colunas numéricas usando StandardScaler."""
    scaler = StandardScaler()
    df[colunas] = scaler.fit_transform(df[colunas])
    return df