import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from pickle import dump
from pathlib import Path

def treinamento_random_forest(df, target_col, nome_arquivo_modelo):
    print(f"\n*** Treinando Random Forest para {nome_arquivo_modelo} ***")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    modelo_base = RandomForestClassifier(random_state=42)

    parametros = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    busca_parametros = GridSearchCV(
        estimator=modelo_base,
        param_grid=parametros,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    busca_parametros.fit(X, y)

    melhor_modelo = RandomForestClassifier(**busca_parametros.best_params_, random_state=42)
    melhor_modelo.fit(X, y)

    resultados_cv = cross_validate(
        melhor_modelo, X, y, cv=10,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    )

    print(f"Accuracy média: {resultados_cv['test_accuracy'].mean():.4f}")
    print(f"Precision média: {resultados_cv['test_precision_macro'].mean():.4f}")
    print(f"Recall médio: {resultados_cv['test_recall_macro'].mean():.4f}")
    print(f"F1-score médio: {resultados_cv['test_f1_macro'].mean():.4f}")

    pasta_modelos = Path(__file__).resolve().parent.parent / "models"
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    caminho_modelo = pasta_modelos / f"modelo_{nome_arquivo_modelo}.pkl"

    with open(caminho_modelo, "wb") as f:
        dump(melhor_modelo, f)

    print(f"Modelo {nome_arquivo_modelo} salvo em {caminho_modelo}")
    return melhor_modelo
