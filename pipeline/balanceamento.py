from imblearn.over_sampling import SMOTE
import pandas as pd

def balanceamento_df(df: pd.DataFrame, coluna_alvo: str) -> pd.DataFrame:
    """
    Aplica balanceamento via SMOTE na coluna alvo.
    Retorna o dataframe balanceado com os dados e a coluna alvo.
    """
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    df_bal = pd.concat([X_bal, y_bal], axis=1)
    print(f"Balanceamento concluído para '{coluna_alvo}': {df_bal.shape[0]} registros.")

    return df_bal
