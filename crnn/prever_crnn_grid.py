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
# 1. DEFINIÇÕES DE CLASSES (Idênticas ao script de treino)
# ======================================================================================
# (As classes CharacterEncoder e CRNN permanecem aqui, sem alterações)
class CharacterEncoder:
    """ Carrega e utiliza o encoder de caracteres treinado. """
    def __init__(self, encoder_path):
        self.encoder = joblib.load(encoder_path)
        self.char_to_int = self.encoder.char_to_int
        self.int_to_char = self.encoder.int_to_char
        self.num_chars = self.encoder.num_chars

    def decode(self, encoded_sequence):
        decoded_chars, last_char_idx = [], -1
        for char_idx in encoded_sequence:
            char_idx = char_idx.item()
            if char_idx == last_char_idx or char_idx == 0:
                last_char_idx = char_idx
                continue
            decoded_chars.append(self.int_to_char[char_idx])
            last_char_idx = char_idx
        return "".join(decoded_chars)

class CRNN(nn.Module):
    """ Arquitetura da Rede Neural (deve ser idêntica à usada no treino). """
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(1024, 256, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.classifier = nn.Linear(256 * 2, num_chars)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)
        return nn.functional.log_softmax(x, dim=2)

# ======================================================================================
# 2. FUNÇÃO DE PREVISÃO (Idêntica à versão anterior)
# ======================================================================================
def predict_single_image(model, image_path, encoder, transform, device):
    try:
        image = Image.open(image_path).convert("L")
    except FileNotFoundError:
        return f"Erro: Imagem '{os.path.basename(image_path)}' não encontrada."
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(image_tensor)
        best_path = torch.argmax(preds, dim=2).squeeze(0)
        predicted_text = encoder.decode(best_path)
    return predicted_text

# ======================================================================================
# 3. BLOCO DE EXECUÇÃO PRINCIPAL (Lógica da Grade 3x3)
# ======================================================================================
if __name__ == '__main__':
    # --- Configuração ---
    MODELO_SALVO_PATH = "crnn_best_model.pth"
    ENCODER_SALVO_PATH = "crnn_char_encoder.joblib"
    CSV_TEST_PATH = "D:/Py/Kaggle_Handwriting_Recognition/written_name_test_v2.csv"
    IMG_TEST_DIR = "D:/Py/Kaggle_Handwriting_Recognition/test"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Carregar Artefatos Salvos ---
    print("\nCarregando modelo, encoder e dados de teste...")
    if not os.path.exists(MODELO_SALVO_PATH) or not os.path.exists(ENCODER_SALVO_PATH):
        print("Erro: Arquivo do modelo ou do encoder não encontrado.")
    else:
        char_encoder = CharacterEncoder(ENCODER_SALVO_PATH)
        num_classes = char_encoder.num_chars
        model = CRNN(num_classes).to(device)
        model.load_state_dict(torch.load(MODELO_SALVO_PATH, map_location=device))
        df_test = pd.read_csv(CSV_TEST_PATH)
        print("Modelo, encoder e dados de teste carregados com sucesso!")

        # --- Preparar Transformação ---
        eval_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # --- Sortear 9 amostras aleatórias do DataFrame de teste ---
        amostras_teste = df_test.sample(9)
        
        # --- Criar a grade de plots 3x3 ---
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Painel de Amostras de Previsão', fontsize=20)

        # Achatando o array de eixos para facilitar a iteração
        axes = axes.flatten()

        for i, (idx, amostra) in enumerate(amostras_teste.iterrows()):
            filename = amostra['FILENAME']
            rotulo_verdadeiro = amostra['IDENTITY']
            caminho_completo = os.path.join(IMG_TEST_DIR, filename)

            # Fazer a previsão
            previsao = predict_single_image(model, caminho_completo, char_encoder, eval_transform, device)

            # Comparar o resultado
            resultado_str = "CORRETO" if str(rotulo_verdadeiro).lower() == previsao.lower() else "INCORRETO"
            cor_titulo = 'green' if resultado_str == "CORRETO" else 'red'

            # Plotar a imagem e o resultado
            ax = axes[i]
            imagem_display = Image.open(caminho_completo)
            ax.imshow(imagem_display, cmap='gray')
            ax.set_title(f"Verdadeiro: '{rotulo_verdadeiro}'\nPrevisto: '{previsao}'\n({resultado_str})", 
                         color=cor_titulo, fontsize=12)
            ax.axis('off')

        # Ajusta o layout para evitar sobreposição e exibe a grade
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()