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
from torch.utils.data import Dataset, DataLoader
import joblib
import matplotlib.pyplot as plt
import Levenshtein  # Para calcular a distância de edição (CER)

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
BEST_MODEL_PATH = "crnn_best_model.pth"
ENCODER_PATH = "crnn_char_encoder.joblib"

# --- Parâmetros de Treinamento ---
# Use um número pequeno para testes rápidos, ou None para usar o dataset completo
NUM_SAMPLES_TO_USE = None  # Começando com um subset para agilidade
BATCH_SIZE = 64
EPOCHS = 200  # Máximo de épocas. O Early Stopping pode parar antes.
LEARNING_RATE = 0.0005
PATIENCE = 5  # Paciência do Early Stopping.

# --- Otimização de Performance ---
NUM_WORKERS = 12  # Comece com 4 para CRNN. Aumente se a GPU ficar ociosa.

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
            if self.verbose: print(f'--> EarlyStopping counter: {self.counter} de {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Salva o modelo quando a perda de validação diminui (checkpoint)."""
        if self.verbose: print(f'--> Perda de validação diminuiu ({self.val_loss_min:.6f} --> {val_loss:.6f}). Salvando modelo...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ======================================================================================
# 4. LÓGICA DE CODIFICAÇÃO DE CARACTERES E DATASET
# ======================================================================================
class CharacterEncoder:
    """Mapeia caracteres para inteiros e vice-versa para a CTC Loss."""
    def __init__(self, vocabulary):
        self.char_to_int = {'-': 0}  # '-' é o caractere 'blank' da CTC no índice 0
        self.int_to_char = {0: '-'}
        for i, char in enumerate(sorted(vocabulary), 1):
            self.char_to_int[char] = i
            self.int_to_char[i] = char
        self.num_chars = len(self.char_to_int)

    def encode(self, text):
        # Filtra caracteres não presentes no vocabulário para evitar erros
        filtered_text = [char for char in str(text).lower() if char in self.char_to_int]
        return [self.char_to_int[char] for char in filtered_text]

    def decode(self, encoded_sequence):
        """ Decodifica uma sequência de previsões da rede, tratando repetições e blanks. """
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

class HandwritingDatasetCTC(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.df.rename(columns={"FILENAME": "image", "IDENTITY": "label"}, inplace=True, errors='ignore')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert("L")
        label = row['label']
        if self.transform: image = self.transform(image)
        return image, label

def collate_fn(batch):
    """Função que agrupa um lote de dados, necessária para labels de tamanhos diferentes."""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, labels

# ======================================================================================
# 5. ARQUITETURA CRNN
# ======================================================================================
class CRNN(nn.Module):
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
# 6. FUNÇÕES AUXILIARES
# ======================================================================================
def calculate_cer(preds, ground_truths, encoder):
    total_distance, total_length = 0, 0
    best_paths = torch.argmax(preds, dim=2).permute(1, 0)
    for i, best_path in enumerate(best_paths):
        decoded_str = encoder.decode(best_path)
        true_str = str(ground_truths[i]).lower()
        if len(true_str) > 0:
            total_distance += Levenshtein.distance(decoded_str, true_str)
            total_length += len(true_str)
    return total_distance, total_length

def train_one_epoch(model, device, train_loader, criterion, optimizer, scaler, encoder):
    model.train()
    total_loss, total_distance, total_length = 0.0, 0, 0
    
    for images, labels in train_loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=='cuda')):
            preds = model(images) # Saída da rede (na GPU)
            
            # Prepara os rótulos e seus tamanhos (na CPU, depois movidos para a GPU)
            text_batch = [encoder.encode(l) for l in labels]
            target_lengths = torch.IntTensor([len(t) for t in text_batch if len(t) > 0])
            targets = torch.cat([torch.IntTensor(t) for t in text_batch if len(t) > 0])
            preds_size = torch.IntTensor([preds.size(0)] * len(labels))
            loss = criterion(preds, targets.to(device), preds_size.to(device), target_lengths.to(device))
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Necessário para o clip_grad_norm funcionar com AMP
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

        if not torch.isnan(loss):
            total_loss += loss.item()
            
        distance, length = calculate_cer(preds.detach(), labels, encoder)
        total_distance += distance
        total_length += length

    avg_loss = total_loss / len(train_loader)
    cer = (total_distance / total_length) * 100 if total_length > 0 else 100
    return avg_loss, cer

def validate_one_epoch(model, device, val_loader, criterion, encoder):
    model.eval()
    total_loss, total_distance, total_length = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=='cuda')):
                preds = model(images)
                text_batch = [encoder.encode(l) for l in labels]
                target_lengths = torch.IntTensor([len(t) for t in text_batch if len(t) > 0])
                targets = torch.cat([torch.IntTensor(t) for t in text_batch if len(t) > 0])
                preds_size = torch.IntTensor([preds.size(0)] * len(labels))
                loss = criterion(preds, targets.to(device), preds_size.to(device), target_lengths.to(device))

            if not torch.isnan(loss):
                total_loss += loss.item()
                
            distance, length = calculate_cer(preds, labels, encoder)
            total_distance += distance
            total_length += length

    avg_loss = total_loss / len(val_loader)
    cer = (total_distance / total_length) * 100 if total_length > 0 else 100
    return avg_loss, cer

def plot_history(history, y_label, title):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label=f'Treino')
    plt.plot(history['val'], label=f'Validação')
    plt.xlabel('Época'), plt.ylabel(y_label), plt.title(title), plt.legend(), plt.grid(True), plt.show()
    
# ======================================================================================
# 7. BLOCO DE EXECUÇÃO PRINCIPAL
# ======================================================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Preparação dos Dados e Encoder ---
    print("Carregando dados e construindo vocabulário...")
    df_train = pd.read_csv(CSV_TRAIN_PATH).dropna()
    df_val = pd.read_csv(CSV_VAL_PATH).dropna()
    df_test = pd.read_csv(CSV_TEST_PATH).dropna()
    
    if NUM_SAMPLES_TO_USE is not None:
        print(f"Usando um subconjunto de {NUM_SAMPLES_TO_USE} amostras de treino para agilidade.")
        df_train = df_train.head(NUM_SAMPLES_TO_USE)
        df_val = df_val.head(int(NUM_SAMPLES_TO_USE * 0.2)) # Usa 20% do subset para validação mais rápida

    full_text = "".join(df_train["IDENTITY"].astype(str).tolist())
    vocabulary = sorted(list(set(full_text.lower())))
    char_encoder = CharacterEncoder(vocabulary)
    num_classes = char_encoder.num_chars
    print(f"Vocabulário criado com {num_classes} classes: {''.join(vocabulary)}")

    # --- Transformações e Datasets ---
    transform = transforms.Compose([transforms.Resize((32, 128)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = HandwritingDatasetCTC(df_train, IMG_TRAIN_DIR, transform=transform)
    val_dataset = HandwritingDatasetCTC(df_val, IMG_VAL_DIR, transform=transform)
    test_dataset = HandwritingDatasetCTC(df_test, IMG_TEST_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    # --- Setup do Modelo e Ferramentas de Treino ---
    model = CRNN(num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True, path=BEST_MODEL_PATH)
    
    # --- Loop de Treinamento ---
    print("\nIniciando treinamento do modelo CRNN...")
    history = {'train_loss': [], 'val_loss': [], 'train_cer': [], 'val_cer': []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_cer = train_one_epoch(model, device, train_loader, criterion, optimizer, scaler, char_encoder)
        val_loss, val_cer = validate_one_epoch(model, device, val_loader, criterion, char_encoder)

        print(f"Época {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train CER: {train_cer:.2f}% | Val Loss: {val_loss:.4f} | Val CER: {val_cer:.2f}%")
        
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_cer'].append(train_cer); history['val_cer'].append(val_cer)
        
        scheduler.step(val_loss)
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("--> Parada antecipada ativada!")
            break

    # --- Finalização do Treino ---
    print(f"\nTreinamento concluído em {(time.time() - start_time)/60:.2f} minutos.")
    print(f"Carregando o melhor modelo salvo de {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    joblib.dump(char_encoder, ENCODER_PATH)
    print(f"Encoder final salvo em: {ENCODER_PATH}")

    # --- Visualização dos Gráficos ---
    plot_history({'train': history['train_loss'], 'val': history['val_loss']}, "Perda (Loss)", "Evolução da Perda (Loss)")
    plot_history({'train': history['train_cer'], 'val': history['val_cer']}, "Taxa de Erro de Caractere (%)", "Evolução do CER (Menor é Melhor)")
    
    # --- Avaliação Final no Conjunto de Teste ---
    print("\n" + "="*50 + "\nAVALIAÇÃO FINAL NO CONJUNTO DE TESTE\n" + "="*50)
    test_loss, test_cer = validate_one_epoch(model, device, test_loader, criterion, char_encoder)
    print(f"Perda Final de Teste: {test_loss:.4f}")
    print(f"Taxa de Erro de Caractere (CER) Final: {test_cer:.2f}%")