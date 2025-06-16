# ======================================================================================
# PAINEL DE PREVISÕES 3x3 COM MODELO CNN DE CLASSIFICAÇÃO
# ======================================================================================

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd
import random

# ======================================================================================
# 1. DEFINIÇÃO DA ARQUITETURA DA REDE (precisa ser idêntica à do treino)
# ======================================================================================
class SimpleCNN(nn.Module):
    """ Arquitetura da Rede Neural Convolucional usada no primeiro modelo. """
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 16, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ======================================================================================
# 2. FUNÇÃO DE PREVISÃO
# ======================================================================================
def predict_single_image_cnn(model, image_path, encoder, transform, device):
    """ Carrega uma imagem, a processa e retorna a previsão do modelo CNN. """
    try:
        image = Image.open(image_path).convert("L")
    except FileNotFoundError:
        return f"Erro: Imagem '{os.path.basename(image_path)}' não encontrada."
        
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        # Pega o índice da classe com a maior probabilidade
        _, pred_index = torch.max(output, 1)
        # Usa o encoder para converter o índice de volta para a palavra
        predicted_label = encoder.inverse_transform([pred_index.item()])
        
    return predicted_label[0]

# ======================================================================================
# 3. BLOCO DE EXECUÇÃO PRINCIPAL (Lógica da Grade 3x3)
# ======================================================================================
if __name__ == '__main__':
    # --- Configuração ---
    # Certifique-se de que os nomes dos arquivos correspondem aos salvos pelo seu primeiro script
    MODELO_SALVO_PATH = "best_model_checkpoint.pth"  # O modelo salvo da CNN
    ENCODER_SALVO_PATH = "label_encoder.joblib" # O encoder de palavras da CNN
    CSV_TEST_PATH = "D:/Py/Kaggle_Handwriting_Recognition/written_name_test_v2.csv"
    IMG_TEST_DIR = "D:/Py/Kaggle_Handwriting_Recognition/test"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Carregar Artefatos Salvos ---
    print("\nCarregando modelo CNN e seu encoder...")
    if not os.path.exists(MODELO_SALVO_PATH) or not os.path.exists(ENCODER_SALVO_PATH):
        print("Erro: Arquivo do modelo ou do encoder não encontrado.")
    else:
        # Carrega o encoder de palavras (LabelEncoder)
        word_encoder = joblib.load(ENCODER_SALVO_PATH)
        num_classes = len(word_encoder.classes_)
        
        # Instancia a arquitetura e carrega os pesos salvos
        model = SimpleCNN(num_classes).to(device)
        model.load_state_dict(torch.load(MODELO_SALVO_PATH, map_location=device))
        
        df_test = pd.read_csv(CSV_TEST_PATH)
        print("✅ Modelo, encoder e dados de teste carregados com sucesso!")

        # --- Preparar Transformação ---
        # Use as mesmas transformações de validação do treinamento da CNN
        eval_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
        ])

        # --- Sortear 9 amostras aleatórias do DataFrame de teste ---
        amostras_teste = df_test.sample(9)
        
        # --- Criar a grade de plots 3x3 ---
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Painel de Amostras de Previsão - Modelo CNN', fontsize=20)
        axes = axes.flatten()

        for i, (idx, amostra) in enumerate(amostras_teste.iterrows()):
            filename = amostra['FILENAME']
            rotulo_verdadeiro = amostra['IDENTITY']
            caminho_completo = os.path.join(IMG_TEST_DIR, filename)

            # Fazer a previsão
            previsao = predict_single_image_cnn(model, caminho_completo, word_encoder, eval_transform, device)

            # Comparar o resultado
            resultado_str = "CORRETO" if str(rotulo_verdadeiro) == str(previsao) else "INCORRETO"
            cor_titulo = 'green' if resultado_str == "CORRETO" else 'red'

            # Plotar a imagem e o resultado
            ax = axes[i]
            imagem_display = Image.open(caminho_completo)
            ax.imshow(imagem_display, cmap='gray')
            ax.set_title(f"Verdadeiro: '{rotulo_verdadeiro}'\nPrevisto: '{previsao}'\n({resultado_str})", 
                         color=cor_titulo, fontsize=12)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()