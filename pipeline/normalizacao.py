from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalizacao_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza as colunas numéricas usando MinMaxScaler.
    """
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Evitar normalizar colunas alvo que são categóricas codificadas (se tiver)
    colunas_para_normalizar = [col for col in colunas_numericas if col not in ['Diagnosis', 'Severity', 'Management']]

    scaler = MinMaxScaler()
    df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])

    print(f"Colunas normalizadas: {colunas_para_normalizar}")
    return df
