import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import joblib
import os
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# ======================================================================================
# 1. DEFINIÇÕES DE CLASSES E FUNÇÕES (Precisam ser idênticas às do script de treino)
# ======================================================================================

class SimpleCNN(nn.Module):
    """ Arquitetura da Rede Neural Convolucional. """
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

class HandwritingDataset(Dataset):
    """ Classe do Dataset Customizado. """
    def __init__(self, df, img_dir, transform=None, encoder=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.encoder = encoder
        
        # Esta linha renomeia a coluna 'IDENTITY' para 'label' no DataFrame
        self.df.rename(columns={"FILENAME": "image", "IDENTITY": "label"}, inplace=True)
        
        self.df['label_enc'] = self.encoder.transform(self.df['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert("L")
        label = row['label_enc']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def evaluate_final(model, device, loader):
    """ Roda a avaliação e retorna a acurácia e perda. """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=='cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, (correct / total) * 100

def predict_single_image(model, image_path, encoder, transform, device):
    """ Faz a previsão para uma única imagem. """
    try:
        image = Image.open(image_path).convert("L")
    except FileNotFoundError:
        return f"Erro: Imagem não encontrada em '{image_path}'"
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, pred_index = torch.max(output, 1)
        predicted_label = encoder.inverse_transform([pred_index.item()])
    return predicted_label[0]

# ======================================================================================
# 2. BLOCO DE EXECUÇÃO PRINCIPAL
# ======================================================================================
if __name__ == '__main__':
    # --- Configuração ---
    MODELO_SALVO_PATH = "best_model_checkpoint.pth"
    ENCODER_SALVO_PATH = "label_encoder.joblib"
    CSV_TEST_PATH = "D:/Py/Kaggle_Handwriting_Recognition/written_name_test_v2.csv"
    IMG_TEST_DIR = "D:/Py/Kaggle_Handwriting_Recognition/test"
    BATCH_SIZE = 128
    NUM_WORKERS = 4 # Use um valor menor para avaliação, pois não há data augmentation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Carregar Artefatos Salvos ---
    print("\nCarregando modelo e encoder salvos...")
    if not os.path.exists(MODELO_SALVO_PATH) or not os.path.exists(ENCODER_SALVO_PATH):
        print("Erro: Arquivo do modelo ou do encoder não encontrado.")
        print("Certifique-se de que os arquivos 'best_model_checkpoint.pth' e 'label_encoder.joblib' estão na mesma pasta.")
    else:
        loaded_encoder = joblib.load(ENCODER_SALVO_PATH)
        num_classes = len(loaded_encoder.classes_)
        
        model = SimpleCNN(num_classes).to(device)
        model.load_state_dict(torch.load(MODELO_SALVO_PATH, map_location=device))
        print("Modelo e encoder carregados com sucesso!")

        # --- Preparar Dados de Teste ---
        df_test = pd.read_csv(CSV_TEST_PATH)
        known_words = set(loaded_encoder.classes_)
        df_test = df_test[df_test['IDENTITY'].isin(known_words)]
        
        eval_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
        ])
        
        # O DataFrame original df_test é passado para a classe
        test_ds = HandwritingDataset(df=df_test, img_dir=IMG_TEST_DIR, transform=eval_transform, encoder=loaded_encoder)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        # --- 1. Executar a Avaliação Final no Conjunto de Teste Completo ---
        print("\nIniciando avaliação final no conjunto de teste...")
        test_loss, test_acc = evaluate_final(model, device, test_loader)
        
        print("\n" + "="*50 + "\n RESULTADO FINAL DA AVALIAÇÃO \n" + "="*50)
        print(f"Acurácia Final no Conjunto de Teste: {test_acc:.2f}%")
        print(f"Perda Final no Conjunto de Teste: {test_loss:.4f}")

        # --- 2. Fazer um Teste Individual com uma Amostra Aleatória ---
        print("\n" + "="*50 + "\n EXEMPLO DE PREVISÃO INDIVIDUAL\n" + "="*50)
        
        # Pega uma amostra aleatória do dataframe de teste
        amostra_teste = df_test.sample(1).iloc[0]
        imagem_nome = amostra_teste['image']
        
        # CORREÇÃO AQUI: Usamos 'label' porque o DataFrame foi renomeado DENTRO da classe HandwritingDataset
        rotulo_verdadeiro = amostra_teste['label']
        
        caminho_completo = os.path.join(IMG_TEST_DIR, imagem_nome)
        
        previsao = predict_single_image(model, caminho_completo, loaded_encoder, eval_transform, device)
        
        if previsao:
            print(f"Arquivo: {imagem_nome}")
            print(f"Rótulo Verdadeiro: {rotulo_verdadeiro}")
            print(f"Previsão do Modelo:  {previsao}")
            
            resultado_str = "CORRETO" if rotulo_verdadeiro == previsao else "INCORRETO"
            print(f"Resultado: {resultado_str}")
            
            imagem_display = Image.open(caminho_completo)
            plt.imshow(imagem_display, cmap='gray')
            plt.title(f"Verdadeiro: {rotulo_verdadeiro}\nPrevisto: {previsao}", color='green' if resultado_str == "CORRETO" else 'red')
            plt.axis('off')
            plt.show()