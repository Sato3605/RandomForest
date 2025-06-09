from joblib import dump

def normalizar_entrada(df_paciente):
    if SCALER is None:
        print("Erro: scaler não está definido.")
        return None
    df_normalizado = SCALER.transform(df_paciente)
    return df_normalizado

def imprimir_metricas(resultados_cv):
    print(f"Recall médio: {resultados_cv['test_recall_macro'].mean():.4f}")
    print(f"F1 Score médio: {resultados_cv['test_f1_macro'].mean():.4f}")

def salvar_modelo(melhor_modelo, nome_arquivo_modelo):
    caminho_modelo = Path(__file__).parent.parent / 'models' / f"{nome_arquivo_modelo}.sav"
    with open(caminho_modelo, 'wb') as arquivo:
        dump(melhor_modelo, arquivo)
    print(f"Modelo {nome_arquivo_modelo} salvo com sucesso em {caminho_modelo}")
