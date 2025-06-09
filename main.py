from pipeline.processamento import import_df, preprocessamento_df
from pipeline.normalizacao import normalizacao_df
from pipeline.balanceamento import balanceamento_df
from treinamento import treinamento_random_forest
from pathlib import Path
from inferencia import coletar_dados_usuario, realizar_inferencia

def criar_pastas_necessarias():
    base = Path(__file__).resolve().parent.parent
    for pasta in ['data', 'models']:
        (base / pasta).mkdir(parents=True, exist_ok=True)

def pipeline_treinamento():
    print(">>> Iniciando processo de treinamento <<<")

    # Importa e trata dados
    print("Importando e preparando dados...")
    dados = import_df()
    dados_tratados = preprocessamento_df(dados)

    # Normaliza
    print("Normalizando dados...")
    dados_norm = normalizacao_df(dados_tratados)
    print("Dados normalizados com sucesso.")

    # Treinamento - Diagnosis
    print("Treinando modelo: Diagnosis")
    dados_diag = dados_norm.copy()
    dados_diag = balanceamento_df(dados_diag, 'Diagnosis')
    treinamento_random_forest(dados_diag, 'Diagnosis', 'diagnosis')

    # Treinamento - Severity (apenas apendicite)
    print("Treinando modelo: Severity")
    dados_sev = dados_norm[dados_norm['Diagnosis'] == 'appendicitis'].copy()
    dados_sev = balanceamento_df(dados_sev, 'Severity')
    treinamento_random_forest(dados_sev, 'Severity', 'severity')

    # Treinamento - Management (apenas apendicite)
    print("Treinando modelo: Management")
    dados_mgmt = dados_norm[dados_norm['Diagnosis'] == 'appendicitis'].copy()
    dados_mgmt = balanceamento_df(dados_mgmt, 'Management')
    treinamento_random_forest(dados_mgmt, 'Management', 'management')

    print(">>> Treinamento finalizado com sucesso <<<")

def menu_principal():
    criar_pastas_necessarias()

    while True:
        print("\n==== Sistema de Diagnóstico Pediátrico ====")
        print("[1] Treinar modelos")
        print("[2] Realizar inferência")
        print("[0] Encerrar programa")

        try:
            escolha = int(input("Escolha uma opção: "))
            if escolha == 1:
                pipeline_treinamento()
            elif escolha == 2:
                dados_paciente = coletar_dados_usuario()
                realizar_inferencia(dados_paciente)
            elif escolha == 0:
                print("Programa encerrado.")
                break
            else:
                print("Opção inválida. Tente novamente.")
        except ValueError:
            print("Entrada inválida! Digite um número.")

if __name__ == "__main__":
    menu_principal()
