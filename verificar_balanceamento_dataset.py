import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÃO ---
CSV_TRAIN_PATH = "D:/Py/Kaggle_Handwriting_Recognition/written_name_train_v2.csv"

print("Analisando o balanceamento do dataset...")

# --- Carregamento e Análise ---
try:
    df = pd.read_csv(CSV_TRAIN_PATH)
    distribuicao_classes = df['IDENTITY'].value_counts()
    
    # --- Apresentação dos Resultados no Console (sem alterações) ---
    print("\n" + "="*50)
    print("📊 ANÁLISE DE BALANCEAMENTO DE CLASSES")
    print("="*50)
    num_total_amostras = len(df)
    num_classes_unicas = len(distribuicao_classes)
    classes_com_uma_amostra = (distribuicao_classes == 1).sum()
    percentual_single = (classes_com_uma_amostra / num_classes_unicas) * 100
    
    print(f"Total de Amostras de Treino: {num_total_amostras}")
    print(f"Total de Classes Únicas (Palavras): {num_classes_unicas}")
    print("\n--- Top 10 Nomes Mais Frequentes ---")
    print(distribuicao_classes.head(10))
    print("\n--- Classes Raras (Cauda Longa) ---")
    print(f"Número de palavras que aparecem APENAS UMA VEZ: {classes_com_uma_amostra}")
    print(f"Isso representa {percentual_single:.2f}% de todas as classes únicas!")

    # --- [NOVO] Visualização Aprimorada com Anotação ---
    print("\nGerando gráfico aprimorado...")
    plt.figure(figsize=(12, 7))
    ax = sns.histplot(distribuicao_classes, log_scale=True, bins=100)
    
    plt.title('Distribuição de Frequência das Classes (Escala Log-Log)', fontsize=20)
    plt.xlabel('Frequência de Ocorrência de uma Classe (Escala Log)', fontsize=18)
    plt.ylabel('Número de Classes com essa Frequência (Escala Log)', fontsize=18)
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # --- Adicionando a anotação para destacar a informação principal ---
    # Texto da anotação
    annotation_text = f'Pico em x=1:\n{classes_com_uma_amostra} classes ({percentual_single:.2f}%)\naparecem apenas uma vez.'
    
    # Adiciona a anotação com uma seta
    ax.annotate(annotation_text, 
                xy=(1, classes_com_uma_amostra), # Ponto exato da seta (x=1, y=contagem)
                xytext=(5, 40000), # Posição do texto
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=20,
                bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", ec="k", lw=1, alpha=0.9))

    plt.tight_layout()
    print("Gráfico gerado. Salve-o para usar no seu artigo.")
    plt.show()

except FileNotFoundError:
    print(f"\nErro: Arquivo não encontrado em '{CSV_TRAIN_PATH}'")