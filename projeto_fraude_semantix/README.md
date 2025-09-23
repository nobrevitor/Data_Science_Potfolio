#  Detecção de Fraudes em Transações Financeiras  

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
- Principais insights:  
  - Transações em horários incomuns apresentam maior chance de fraude.  
  - Localizações divergentes entre compras sucessivas são fortes indícios suspeitos.  

*(Insira aqui gráficos salvos em `images/` para ilustrar resultados e métricas)*  

---

## ✅ Conclusões e Próximos Passos  
- O modelo é aplicável em sistemas de monitoramento de transações financeiras.  
- Próximos passos:  
  - Implantar em **API** para predição em tempo real.  
  - Reduzir ainda mais os **falsos positivos**.  
  - Explorar modelos de **deep learning** para padrões complexos.  

---

## 🗂 Estrutura do Repositório  

