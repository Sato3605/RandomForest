import pandas as pd

def import_df(caminho_arquivo='data/regensburg_pediatric_appendicitis.csv'):
    """
    Importa os dados da base CSV.
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        print(f"Dados importados com sucesso: {df.shape[0]} registros.")
        return df
    except FileNotFoundError:
        print("Arquivo não encontrado. Verifique o caminho.")
        return pd.DataFrame()

def preprocessamento_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processamento básico: limpeza, tratamento de missing, encoding e conversões.
    Adapte conforme sua base.
    """
    # Exemplo: remover colunas irrelevantes (adicione as colunas que quiser remover)
    colunas_relevantes = [
        'Age', 'BMI', 'Sex', 'Symptom1', 'Symptom2', # exemplo
        'Diagnosis', 'Severity', 'Management'
    ]
    df = df[colunas_relevantes].copy()

    # Remover registros com valores nulos
    df.dropna(inplace=True)

    # Encoding simples de sexo: M=1, F=0
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

    # Se houver mais codificações, faça aqui (exemplo para diagnosis, severity, management)
    # Pode usar label encoding para as colunas alvo no treinamento

    print(f"Dados pré-processados: {df.shape[0]} registros após limpeza.")
    return df
