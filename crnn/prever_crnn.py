import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import os
import matplotlib.pyplot as plt

# ======================================================================================
# 1. DEFINIÇÕES DE CLASSES (Precisam ser idênticas às do script de treino)
# ======================================================================================

class CharacterEncoder:
    """ Carrega e utiliza o encoder de caracteres treinado. """
    def __init__(self, encoder_path):
        # Carrega o encoder salvo com joblib
        self.encoder = joblib.load(encoder_path)
        self.char_to_int = self.encoder.char_to_int
        self.int_to_char = self.encoder.int_to_char
        self.num_chars = self.encoder.num_chars

    def decode(self, encoded_sequence):
        """ Decodifica uma sequência de previsões da rede. """
        decoded_chars = []
        last_char_idx = -1
        for char_idx in encoded_sequence:
            char_idx = char_idx.item()
            if char_idx == last_char_idx or char_idx == 0:  # Ignora repetição ou blank
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
# 2. FUNÇÃO DE PREVISÃO
# ======================================================================================
def predict_single_image(model, image_path, encoder, transform, device):
    """Carrega uma imagem, a processa e retorna a previsão do modelo."""
    try:
        image = Image.open(image_path).convert("L")
    except FileNotFoundError:
        print(f"Erro: Imagem não encontrada em '{image_path}'")
        return None

    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(image_tensor)
        # Pega o melhor caminho (best path) da matriz de probabilidades
        best_path = torch.argmax(preds, dim=2).squeeze(0)
        # Decodifica a sequência de índices para texto
        predicted_text = encoder.decode(best_path)
        
    return predicted_text

# ======================================================================================
# 3. BLOCO DE EXECUÇÃO DA PREVISÃO
# ======================================================================================
if __name__ == '__main__':
    # --- Configuração ---
    MODELO_SALVO_PATH = "crnn_best_model.pth"
    ENCODER_SALVO_PATH = "crnn_char_encoder.joblib"
    
    # Caminho da imagem a ser testada
    IMAGEM_PARA_TESTAR = "D:/Py/Kaggle_Handwriting_Recognition/testes_tiago/tiago.jpeg" 

    # --- Setup do dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Carregar Artefatos Salvos ---
    print("\nCarregando modelo e encoder salvos...")
    if not os.path.exists(MODELO_SALVO_PATH) or not os.path.exists(ENCODER_SALVO_PATH):
        print("Erro: Arquivo do modelo ou do encoder não encontrado.")
        print("Certifique-se de que os arquivos 'crnn_best_model.pth' e 'crnn_char_encoder.joblib' estão na mesma pasta que este script.")
    else:
        # Carrega o encoder
        char_encoder = CharacterEncoder(ENCODER_SALVO_PATH)
        num_classes = char_encoder.num_chars
        
        # Instancia a arquitetura e carrega os pesos
        model = CRNN(num_classes).to(device)
        model.load_state_dict(torch.load(MODELO_SALVO_PATH, map_location=device))
        
        print("Modelo e encoder carregados com sucesso!")

        # Define as mesmas transformações de avaliação do treinamento
        eval_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # --- Fazer a Previsão ---
        previsao = predict_single_image(model, IMAGEM_PARA_TESTAR, char_encoder, eval_transform, device)

        if previsao is not None:
            print("\n" + "="*50 + "\n🔍 Resultado da Previsão\n" + "="*50)
            print(f"    Arquivo: {IMAGEM_PARA_TESTAR}")
            print(f"    Texto Previsto pelo Modelo: '{previsao}'")
            
            # Mostra a imagem para comparação visual
            try:
                imagem_display = Image.open(IMAGEM_PARA_TESTAR)
                plt.imshow(imagem_display, cmap='gray')
                plt.title(f"Previsão: {previsao}")
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Não foi possível exibir a imagem. Erro: {e}")