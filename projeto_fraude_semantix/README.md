#  Detecção de Fraudes em Transações Financeiras  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=plotly)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)](https://seaborn.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EB5E00)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Demo-FF4B4B?logo=streamlit)](https://streamlit.io/)

_Modelo de Machine Learning para identificar transações fraudulentas em tempo real._  

---

## 📌 Contexto  
A fraude em cartões de crédito e transações financeiras é um dos maiores desafios do setor bancário, causando bilhões em perdas todos os anos.  
Com o aumento do comércio eletrônico e pagamentos digitais, é essencial identificar rapidamente atividades suspeitas para proteger os clientes e as instituições.  

Este projeto tem como objetivo construir um modelo de classificação capaz de detectar fraudes com alta precisão e baixo número de falsos positivos.  

---

## 🎯 Objetivos  
- **Analisar** padrões de comportamento entre transações legítimas e fraudulentas.  
- **Desenvolver** modelos de machine learning para classificar transações.  
- **Avaliar** métricas críticas, como *precision*, *recall* e *F1-score*, focando na detecção de fraudes.  
- **Gerar insights** que auxiliem na criação de políticas antifraude.  

---

## 📊 Dataset  
- **Fonte**: [Kaggle – Fraud Detection Transactions Dataset](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset/data)
- **Número de registros**: 50000  
- **Número de variáveis**: 21  
- **Variável alvo**:  
  - `Fraud_Label` → 1 (fraude), 0 (legítima)  
  
---

## 🔎 Metodologia  
1. **Análise Exploratória de Dados (EDA)**  
   - Distribuição das covariáveis.  
   - Diferenças entre fraudes e não fraudes.  

2. **Preparação dos Dados**  
   - Tratamento de dados ausentes.  
   - Feature Engineering.  
   - Balanceamento de classes (SMOTE).  

3. **Modelagem**  
   - Algoritmos testados:  
     - Random Forest  
     - XGBoost  
   - Ajuste de hiperparâmetros com GridSearch.  

4. **Avaliação**  
   - **Métricas principais**: Recall, Precision, F1-Score, AUC-ROC.  
   - Matriz de confusão para avaliar erros.  

---

## 📈 Resultados  
- O modelo XGBoost apresentou o melhor equilíbrio entre **recall** e **precisão**.
- Após o tunning o modelo passou a detectar mais fraudes e reduziu o valor de falsos negativos.
- Resultados das métricas:
  - Classification Report:

              precision    recall  f1-score   support

             0      0.83      0.88      0.86      7602
             1      0.72      0.63      0.67      3636

   accuracy                             0.80     11238
   macro avg        0.78      0.76      0.76     11238
   weighted avg     0.80      0.80      0.80     11238

ROC-AUC: 0.7915
Log Loss: 0.4485  
- Principais insights:  
  - Número maior de transações falhas sugerem maior risco de fraude.  
  - Apesar dos riscos associados ao uso de caixas eletrônicos (ATMs) e de dispositivos eletrônicos com baixa segurança, o maior fator de vulnerabilidade está no comportamento do próprio usuário, que muitas vezes acaba criando brechas em qualquer sistema de proteção.

---

## 🗂 Estrutura do Repositório  


