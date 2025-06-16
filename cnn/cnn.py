# ======================================================================================
# 1. IMPORTS E SETUP INICIAL
# ======================================================================================
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import joblib
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# Garante que o Matplotlib não cause problemas com múltiplos processos no Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ======================================================================================
# 2. CONFIGURAÇÃO GLOBAL
# ======================================================================================
# --- Caminhos para os arquivos ---
CSV_TRAIN_PATH = "D:/Py/Kaggle_Handwriting_Recognition/written_name_train_v2.csv"
CSV_VAL_PATH = "D:/Py/Kaggle_Handwriting_Recognition/written_name_validation_v2.csv"
CSV_TEST_PATH = "D:/Py/Kaggle_Handwriting_Recognition/written_name_test_v2.csv"
IMG_TRAIN_DIR = "D:/Py/Kaggle_Handwriting_Recognition/train"
IMG_VAL_DIR = "D:/Py/Kaggle_Handwriting_Recognition/validation"
IMG_TEST_DIR = "D:/Py/Kaggle_Handwriting_Recognition/test"

# --- Arquivos de Saída ---
BEST_MODEL_PATH = "best_model_checkpoint.pth"
ENCODER_PATH = "label_encoder.joblib"

# --- Parâmetros de Treinamento ---
# Use None para treinar com o dataset completo. Coloque um número para um teste rápido.
NUM_SAMPLES_TO_USE = None 
BATCH_SIZE = 64
EPOCHS = 100  # Máximo de épocas. O Early Stopping pode parar antes.
LEARNING_RATE = 0.001
PATIENCE = 5  # Paciência do Early Stopping.

# --- Otimização de Performance ---
# 8 tive problemas com arquivo de paginação do Windows... :(
NUM_WORKERS = 4

# ======================================================================================
# 3. CLASSE DE EARLY STOPPING
# ======================================================================================
class EarlyStopping:
    """Interrompe o treinamento se a perda de validação não melhorar."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'--> EarlyStopping counter: {self.counter} de {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Salva o modelo quando a perda de validação diminui (checkpoint)."""
        if self.verbose:
            print(f'--> Perda de validação diminuiu ({self.val_loss_min:.6f} --> {val_loss:.6f}). Salvando modelo...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ======================================================================================
# 4. CLASSES DO DATASET E MODELO
# ======================================================================================
class HandwritingDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, encoder=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.encoder = encoder
        
        # Garante que o encoder foi treinado antes de ser usado
        if not hasattr(self.encoder, 'classes_'):
             raise RuntimeError("Encoder must be fitted before transforming labels.")
        
        self.df.columns = [col.strip().upper() for col in self.df.columns]
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

class SimpleCNN(nn.Module):
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
# 5. FUNÇÕES AUXILIARES
# ======================================================================================
def train_one_epoch(model, device, train_loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0, 0, 0
    # Define o tipo de dispositivo para o autocast ('cuda' ou 'cpu')
    device_type = device.type

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        # NOVA API de autocast: mais geral e preparada para o futuro
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type=='cuda')):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        # A lógica do otimizador agora diferencia se o scaler está sendo usado
        if scaler: # Se estiver na GPU, scaler não será None
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # Lógica para CPU (sem scaler)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def validate_one_epoch(model, device, val_loader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    device_type = device.type
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # NOVA API de autocast
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type=='cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def plot_history(train_values, val_values, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train_values, label=f'Treino')
    plt.plot(val_values, label=f'Validação')
    plt.xlabel('Época')
    plt.ylabel(title)
    plt.title(f'{title} por Época')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_single_image(model, image_path, encoder, transform, device):
    try:
        image = Image.open(image_path).convert("L")
    except FileNotFoundError:
        return f"Erro: Imagem não encontrada em {image_path}"
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, pred_index = torch.max(output, 1)
        predicted_label = encoder.inverse_transform([pred_index.item()])
    return predicted_label[0]

# ======================================================================================
# 6. BLOCO DE EXECUÇÃO PRINCIPAL
# ======================================================================================
def main():
    # --- Setup do dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Carregamento e Preparação dos Dados ---
    print("\nCarregando e preparando os dados...")
    df_train_full = pd.read_csv(CSV_TRAIN_PATH)
    df_val = pd.read_csv(CSV_VAL_PATH)
    df_test = pd.read_csv(CSV_TEST_PATH)

    encoder = LabelEncoder()
    encoder.fit(df_train_full['IDENTITY'])
    num_classes = len(encoder.classes_)
    print(f"Encoder criado. Número de classes: {num_classes}")

    known_words = set(encoder.classes_)
    df_val = df_val[df_val['IDENTITY'].isin(known_words)]
    df_test = df_test[df_test['IDENTITY'].isin(known_words)]

    # --- Definição das Transformações ---
    train_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.RandomAffine(degrees=5, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
    ])

    # --- Criação dos Datasets ---
    train_ds_full = HandwritingDataset(df=df_train_full, img_dir=IMG_TRAIN_DIR, transform=train_transform, encoder=encoder)
    val_ds = HandwritingDataset(df=df_val, img_dir=IMG_VAL_DIR, transform=eval_transform, encoder=encoder)
    test_ds = HandwritingDataset(df=df_test, img_dir=IMG_TEST_DIR, transform=eval_transform, encoder=encoder)
    
    # --- Seleção do Dataset de Treino (Completo ou Subconjunto) ---
    training_dataset = Subset(train_ds_full, range(min(NUM_SAMPLES_TO_USE, len(train_ds_full)))) if NUM_SAMPLES_TO_USE is not None else train_ds_full
    print(f"Tamanho do dataset de treino: {len(training_dataset)} amostras.")
        
    # --- Criação dos DataLoaders Otimizados ---
    train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False)
    
    # --- Setup do Modelo e Treinamento ---
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # NOVA API de GradScaler: criado apenas se CUDA estiver disponível
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True, path=BEST_MODEL_PATH)
    
    # --- Loop de Treinamento ---
    print("\nIniciando o treinamento...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate_one_epoch(model, device, val_loader, criterion)

        print(f"Época {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("--> Parada antecipada ativada!")
            break

    total_time = time.time() - start_time
    print(f"\nTreinamento concluído em {total_time/60:.2f} minutos.")

    # --- Carrega o melhor modelo e salva o encoder ---
    print(f"Carregando o melhor modelo salvo do checkpoint: {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Encoder final salvo em: {ENCODER_PATH}")

    # --- Visualização dos Resultados do Treino ---
    plot_history(history['train_acc'], history['val_acc'], "Acurácia")
    plot_history(history['train_loss'], history['val_loss'], "Perda (Loss)")

    # --- Avaliação Final no Conjunto de Teste ---
    print("\n" + "="*50 + "\nAVALIAÇÃO FINAL NO CONJUNTO DE TESTE\n" + "="*50)
    test_loss, test_acc = validate_one_epoch(model, device, test_loader, criterion)
    print(f"Acurácia Final no Conjunto de Teste: {test_acc:.2f}%")
    print(f"Perda Final no Conjunto de Teste: {test_loss:.4f}")

    # --- Exemplo de Previsão Individual ---
    print("\n" + "="*50 + "\nEXEMPLO DE PREVISÃO INDIVIDUAL\n" + "="*50)
    if not df_test.empty:
        amostra_teste = df_test.sample(1).iloc[0]
        imagem_nome = amostra_teste['image']
        rotulo_verdadeiro = amostra_teste['label']
        caminho_completo = os.path.join(IMG_TEST_DIR, imagem_nome)

        previsao = predict_single_image(model, caminho_completo, encoder, eval_transform, device)
        print(f"Arquivo: {imagem_nome}")
        print(f"Rótulo Verdadeiro: {rotulo_verdadeiro}")
        print(f"Previsão do Modelo:  {previsao}")
        
        resultado_str = "CORRETO" if rotulo_verdadeiro == previsao else "INCORRETO"
        print(f"Resultado: {resultado_str}")

# Ponto de entrada do script - ESSENCIAL para num_workers no Windows
if __name__ == '__main__':
    main()