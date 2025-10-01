# 📊 Data Science Portfolio

Portfólio de projetos desenvolvidos em Ciência de Dados, Machine Learning e Estatística, com aplicações práticas em diferentes contextos de previsão e análise de dados.

---

## Detecção de Fraudes em Transações Financeiras

📌 **Descrição:**

Com o aumento das transações digitais, a segurança financeira tornou-se essencial. Neste projeto, desenvolvi um modelo de classificação de fraudes para identificar transações suspeitas em tempo real, utilizando técnicas de análise estatística e machine learning.

📌 **Principais etapas:**

- Análise exploratória para identificar padrões de fraude.
- Engenharia de variáveis, como taxa de transações falhas e perfis de autenticação.
- Criação e avaliação de modelos de classificação (XGBoost e RandonForest).
- Métricas utilizadas: ROC-AUC, Precision, Recall e Log Loss.
- Geração de insights para reforçar a segurança de clientes e instituições.

📌 **Impacto:**
O modelo oferece suporte à detecção precoce de fraudes e auxilia na redução de perdas financeiras, contribuindo para sistemas mais seguros.

📂 **Pasta:** `projeto_fraude_semantix`

---

## Previsão de Renda

📌 **Descrição:**

Desenvolvi um modelo preditivo para estimar a renda de clientes, aplicando machine learning supervisionado e técnicas de seleção de variáveis.

📌 **Principais etapas:**

- Análise exploratória e descritiva dos dados (EDA).
- Tratamento de missing values e outliers (capping pelo IQR e log-transformação).
- Seleção de variáveis via LassoCV e Permutation Importance.
- Modelagem com Gradient Boosting Regressor, otimizado com Optuna.
- Avaliação com RMSE, MAE e R², incluindo deslogarização para interpretação.

📌 **Ferramentas:** Python, Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn, Optuna.

📂 **Pasta:** `Previsão de Renda`

---

## Modelagem de Regressão Linear

📌 **Descrição:**

Criação de um modelo de regressão linear para previsão de renda, com foco em análise exploratória e seleção de variáveis.

📌 **Principais etapas:**

- EDA e tratamento de dados.
- Seleção de variáveis significativas com Statmodels (LASSO).
- Criação, treino, ajuste e avaliação do modelo com Scikit-Learn.

📌 **Ferramentas:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Statmodels.

📂 **Pasta:** `Modelagem de Regressão Linear`

---

## Árvore de Regressão

📌 **Descrição:**

Modelo preditivo para estimar o preço de casas com base em variáveis explicativas.

📌 **Principais etapas:**

- EDA com Pandas, Seaborn e Matplotlib.
- Modelagem com DecisionTreeRegressor (Scikit-Learn).
- Ajustes com base no melhor ccp_alpha (pruning).
- Avaliação com métricas MSE e R².

📌 **Ferramentas:** Python, Pandas, Matplotlib, Seaborn, Scikit-Learn.

📂 **Pasta:** `Árvore de Regressão`

---

## 🛠️ Tecnologias Utilizadas

**Linguagens:** Python, SQL

**Bibliotecas:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, SciPy, XGBoost, Optuna, Statmodels, Imbalanced-Learn, Shap, Feature-Engine

**Ferramentas:** Jupyter Notebook, Streamlit, Power BI, PostgreSQL, Git/GitHub

**Áreas:** Ciência de Dados, Machine Learning, Estatística, Modelagem Preditiva, Análise Exploratória de Dados
