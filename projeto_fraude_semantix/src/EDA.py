import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def univariada(df, var: str, target=None):
    """
    Plota a distribuição de uma variável do DataFrame, diferenciando o tipo (numérica ou categórica)
    e o número de categorias, oferecendo visualizações adaptadas para melhor análise exploratória.

    Comportamento:

    1. Variáveis numéricas:
       - Exibe um boxplot na parte superior para identificar outliers e distribuição geral.
       - Exibe um histograma com KDE na parte inferior para analisar frequência e forma da distribuição.

    2. Variáveis categóricas:
       - < 20 categorias: gráfico de barras vertical com rótulos de contagem em cada barra.
       - 20 a 30 categorias: gráfico de barras horizontal ordenado, com rótulos de contagem ao lado.
       - > 30 categorias: seleciona 5 categorias representativas 
         (máxima, intermediário alto, mediana, intermediário baixo e mínima) e plota gráfico de barras vertical
         mostrando a contagem apenas dessas categorias, com rótulos.

    3. Target (opcional):
       - Se fornecido:
         * Numéricas: exibe a média da variável para cada classe do target.
         * Categóricas: exibe a proporção do target em cada categoria (em %).

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    var : str
        Nome da variável a ser analisada.
    target : str, opcional
        Nome da variável target binária (ex.: "Is_Fraud"). Se fornecida, exibe proporções junto ao gráfico.

    Observações:
    -------------
    - Valores nulos são ignorados na análise.
    - Para variáveis categóricas com muitas categorias, a seleção das 5 categorias representativas
      permite visualizar rapidamente os extremos e a mediana sem sobrecarregar o gráfico.
    - As cores são definidas de forma padronizada para manter consistência visual entre gráficos.
    - Labels de contagem são adicionadas automaticamente em todas as barras visíveis.

    Exemplo de uso:
    ---------------
    >>> univariada(df, "idade")
    >>> univariada(df, "categoria_produto", target="Is_Fraud")
    """
    
    serie = df[var].dropna()
    df_temp = df.copy()
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if pd.api.types.is_numeric_dtype(serie):
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
        
        cor = sns.color_palette('crest')[2]
        
        sns.boxplot(x=serie, ax=ax[0], color=cor)
        ax[0].set(xlabel='')
        ax[0].grid(True, linestyle='--', alpha=0.4)

        sns.histplot(serie, kde=True, ax=ax[1], color=cor)
        ax[1].set_xlabel(var, labelpad=10)
        ax[1].set_ylabel('Frequência')

        plt.suptitle(f"Distribuição - {var}", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

        if target is not None:
            df_temp[target] = df_temp[target].map({1: 'Fraude', 0: 'Genuina'})
            medio = df_temp.groupby(target)[var].mean().reset_index(name=f'media_{var}')
            medio = medio.set_index(target)
            print(medio.round(2))

    else:
        counts = serie.value_counts()
        n_cats = len(counts)

        if n_cats < 20:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                x=counts.index, 
                y=counts.values, 
                hue=counts.index, 
                palette='viridis', 
                dodge=False, 
                legend=False
            )
            plt.xlabel(var, labelpad=10)
            plt.ylabel('Contagem')
            plt.title(f"Distribuição - {var}")
            plt.grid(True, linestyle='--', alpha=0.4)

            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()
            
            if target is not None:
                prop = df.groupby([var, target]).size().unstack(fill_value=0)
                prop.columns = ['Genuina', 'Fraude']
                prop_percentual = prop.div(prop.sum(axis=1), axis=0)
                prop_formatado = prop_percentual.map(lambda x: f'{x:.2%}')

                print(prop_formatado.sort_values(by='Fraude', ascending=False))

        elif 20 <= n_cats <= 31:
            counts_sorted = counts.sort_values(ascending=False)
            plt.figure(figsize=(10, 8))
            ax = sns.barplot(
                y=counts_sorted.index, 
                x=counts_sorted.values, 
                hue=counts_sorted.index,
                palette='viridis', 
                dodge=False, 
                legend=False
            )
            plt.xlabel('Contagem', labelpad=10)
            plt.ylabel(var)
            plt.title(f"Distribuição - {var}")
            plt.grid(True, linestyle='--', alpha=0.4)

            for p in ax.patches:
                width = p.get_width()
                ax.annotate(f'{int(width)}',
                            (width, p.get_y() + p.get_height() / 2.),
                            ha='left', va='center', fontsize=9)

            plt.tight_layout()
            plt.show()
            
            if target is not None:
                prop = df.groupby([var, target]).size().unstack(fill_value=0)
                prop.columns = ['Genuina', 'Fraude']
                prop_percentual = prop.div(prop.sum(axis=1), axis=0)
                prop_formatado = prop_percentual.map(lambda x: f'{x:.2%}')

                print(prop_formatado.sort_values(by='Fraude', ascending=False))

        else:
            sorted_counts = counts.sort_values(ascending=False)
            max_cat = sorted_counts.index[0]
            median_cat = sorted_counts.index[n_cats // 2]
            min_cat = sorted_counts.index[-1]
            mid_high_cat = sorted_counts.index[n_cats // 4]
            mid_low_cat = sorted_counts.index[3 * n_cats // 4]

            selected_cats = [max_cat, mid_high_cat, median_cat, mid_low_cat, min_cat]
            selected_counts = sorted_counts[selected_cats]
            df_filtrado = df[df[var].isin(selected_cats)]

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
            x=selected_counts.index, 
            y=selected_counts.values, 
            hue=selected_counts.index, 
            palette='viridis', 
            dodge=False, 
            legend=False
            )
            plt.xlabel(var, labelpad=10)
            plt.ylabel('Contagem')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.title(f"Distribuição (5 categorias representativas) - {var}")
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            plt.show()
            
            if target is not None:
                prop = df_filtrado.groupby([var, target]).size().unstack(fill_value=0)
                prop.columns = ['Genuina', 'Fraude']
                prop_percentual = prop.div(prop.sum(axis=1), axis=0)
                prop_formatado = prop_percentual.map(lambda x: f'{x:.2%}')

                print(prop_formatado.sort_values(by='Fraude', ascending=False))

def estabilidade_tempo(df, var, date_col, freq='D'):
    '''
    Plota a estabilidade de uma variável ao longo do tempo.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame original.
    var : str
        Nome da variável a ser analisada.
    date_col : str
        Nome da coluna de datas.
    freq : str, opcional
        Frequência de agregação ('D' - dia, 'W' - semana, 'M' - mês). Default é diário.

    Observações:
    -------------
    - O DataFrame original não é alterado.
    - A função converte a coluna de datas para datetime se necessário.
    - Para variáveis binárias, o gráfico mostra proporção (%) e o eixo y é ajustado.
    '''

    df_aux = df[[var, date_col]].copy()

    if not pd.api.types.is_datetime64_any_dtype(df_aux[date_col]):
        df_aux[date_col] = pd.to_datetime(df_aux[date_col], errors='coerce')

    is_binary = df_aux[var].nunique() == 2

    df_time = df_aux.groupby(pd.Grouper(key=date_col, freq=freq))[var].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_time, x=date_col, y=var, marker='o')

    if is_binary:
        plt.ylabel(f'Proporção de {var}')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.ylim(0, df_time[var].max() * 1.1) 
    else:
        plt.ylabel(f'Média de {var}')

    plt.title(f'Estabilidade de {var} ao longo do tempo')
    plt.xlabel('Tempo')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def correlacao(df):
    """
    Aplica LabelEncoder nas variáveis categóricas e plota um heatmap de correlação.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com variáveis numéricas e categóricas.
    """
    
    df_encoded = df.copy()
    
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    corr = df_encoded.corr()
    
    plt.figure(figsize=(16,12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'label':'Correlação'})
    plt.title('Mapa de Correlação - Variáveis Numéricas e Categóricas Codificadas', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()