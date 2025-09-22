import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import numpy as np
from feature_engine.encoding import CountFrequencyEncoder

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
        if serie.nunique() > 2:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
            
            discrete = True if 4 < serie.nunique() < 15 else False
            kde = False if discrete == True else True
            
            cor = sns.color_palette('crest')[2]
            
            sns.boxplot(x=serie, ax=ax[0], color=cor)
            ax[0].set(xlabel='')
            ax[0].grid(True, linestyle='--', alpha=0.4)

            sns.histplot(serie, kde=kde, ax=ax[1], color=cor, discrete=discrete)
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
            counts = df[var].value_counts().sort_index()
            
            plt.figure(figsize=(10,6))
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
        Frequência de agregação ('D' - dia, 'W' - semana, 'ME' - mês). Default é diário.

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


def correlacao(df, target):
    """
    Aplica CountFrequencyEncoder nas variáveis categóricas e plota um heatmap de correlação,
    colocando a variável alvo nos cantos (última coluna/linha).

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com variáveis numéricas e categóricas.
    target : str
        Nome da variável alvo.
    """
    
    df_encoded = df.copy()
    var_cat = df_encoded.select_dtypes(include=['object', 'category']).columns.to_list()
    
    if len(var_cat) != 0:
    
        # Seleciona variáveis categóricas
        var_cat = df_encoded.select_dtypes(include=['object', 'category']).columns.to_list()

        # Codifica as categóricas
        encoder = CountFrequencyEncoder(encoding_method='frequency',
                                        variables=var_cat,
                                        ignore_format=True)
        df_encoded = encoder.fit_transform(df_encoded)


    # Reordena colunas para colocar o target no final
    cols = [col for col in df_encoded.columns if col != target] + [target]
    df_encoded = df_encoded[cols]
    
    # Calcula correlação
    corr = df_encoded.corr()
    
    # Plota heatmap
    plt.figure(figsize=(16,12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'label':'Correlação'})
    plt.title('Mapa de Correlação - Variáveis Numéricas e Categóricas Codificadas', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
def WOE(df: pd.DataFrame, feature: str, target: str, top_n: int = 20):
    '''
    Plota o WOE por categoria de uma variável.
    
    Parâmetros
    ----------
    df : DataFrame
        Base de dados.
    feature : str
        Nome da variável que será analisada.
    target : str
        Variável alvo binária (0 = evento, 1 = não evento).
    top_n : int
        Máximo de categorias a mostrar no gráfico (restante vai para 'Outros').
    '''

    df_temp = df[[feature, target]].copy()

    if pd.api.types.is_numeric_dtype(df_temp[feature]) and df_temp[feature].nunique() > 10:
        try:
            df_temp[feature] = pd.qcut(df_temp[feature], q=10, duplicates='drop')
        except:
            df_temp[feature] = pd.cut(df_temp[feature], bins=10)

    tab = pd.crosstab(df_temp[feature], df_temp[target])
    tab = tab + 0.0001

    pct_evento = tab[0] / tab[0].sum()
    pct_nao_evento = tab[1] / tab[1].sum()

    woe = np.log(pct_evento / pct_nao_evento)
    woe_df = woe.reset_index()
    woe_df.columns = [feature, 'WOE']

    woe_df = woe_df.reindex(woe_df.WOE.abs().sort_values(ascending=False).index)

    if len(woe_df) > top_n:
        top = woe_df.iloc[:top_n]
        resto = pd.DataFrame({
            feature: ['Outros'],
            'WOE': [woe_df.iloc[top_n:]['WOE'].mean()]
        })
        woe_df = pd.concat([top, resto])

    plt.figure(figsize=(10, 6))
    plt.barh(woe_df[feature].astype(str), woe_df['WOE'], color=sns.color_palette('crest')[2], edgecolor=None)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f'WOE por categoria - {feature}', fontsize=14)
    plt.xlabel('WOE', fontsize=12)
    plt.ylabel('Categoria', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    
def IV(df: pd.DataFrame, target: str, limiar_qcut=10) -> pd.DataFrame:
    '''
    Calcula o Information Value (IV) das variáveis explicativas em relação a uma variável target binária.

    O Information Value é uma métrica usada para medir a capacidade preditiva de uma variável explicativa 
    em problemas de classificação binária, sendo muito utilizada em análise de crédito e modelagem de risco.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados das variáveis explicativas e target.
    - target (str): Nome da variável target binária (deve conter apenas duas classes).
    - limiar_qcut (10, opcional): Limiar para categorização de variáveis com mais de 10 valores únicos (padrão: 10).
    
    Descrição:
    - Recebe um DataFrame com as variáveis explicativas (`df`) e a variável target binária (`target`).
    - Se a variável explicativa for numérica e possuir mais de 10 valores únicos, ela é categorizada em 10 bins por quantis.
    - Calcula a tabela cruzada de frequências entre as categorias de `feature` e as classes de `target`.
    - Aplica um ajuste para evitar divisão por zero somando 0.00001 às frequências.
    - Calcula o Weight of Evidence (WoE) para cada categoria.
    - Calcula e retorna o Information Value (IV) agregando o impacto das categorias.

    Retorno:
    - pd.DataFrame: Contendo o valor do Information Value para cada variável explicativa do DataFrame.

    Observações:
    - A variável target deve ser binária, com duas classes distintas.
    - A discretização é feita automaticamente para variáveis numéricas com mais de 10 valores únicos,
      mas pode ser ajustada conforme a necessidade.
    '''
    results = []
    for feature in df.drop(columns=[target]).columns:
        df_temp = df[[feature, target]].copy()

        if pd.api.types.is_numeric_dtype(df_temp[feature]):
            num_unique = df_temp[feature].nunique()
        
            if num_unique > limiar_qcut:
                try:
                    df_temp[feature] = pd.qcut(df_temp[feature], q=10, duplicates='drop')
                except:
                    pass
        
        tab = pd.crosstab(df_temp[feature], df_temp[target])
        tab = tab + 0.0001

        col_evento, col_nao_evento = tab.columns[1], tab.columns[0]

        pct_evento = tab[col_evento] / tab[col_evento].sum()
        pct_nao_evento = tab[col_nao_evento] / tab[col_nao_evento].sum()

        woe = np.log(pct_evento / pct_nao_evento)
        iv = np.sum((pct_evento - pct_nao_evento) * woe)
        
        results.append({
            'Variavel': feature,
            'IV': iv.round(4)
        })

    return pd.DataFrame(results).sort_values(by='IV', ascending=False).reset_index(drop=True)