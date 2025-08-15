import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier

def treinar_isolation_forest(df, colunas):
    modelo = IsolationForest(contamination=0.02, random_state=42)
    modelo.fit(df[colunas])
    return modelo

def treinar_classificador_rf(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def salvar_modelo(modelo, caminho):
    joblib.dump(modelo, caminho)

def carregar_modelo(caminho):
    return joblib.load(caminho)