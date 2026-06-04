import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# 1. AUTOENCODER (AE) MİMARİSİ
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder (Sıkıştırma Bölümü)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8), # Darboğaz (Bottleneck) Katmanı 16'dan 8'e düşürüldü
            nn.ReLU(True)
        )
        
        # Decoder (Geri Oluşturma Bölümü)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Girdi verimiz Min-Max Scaling ile 0-1 arasında olduğu için Sigmoid kullanıyoruz
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder():
    klasor = "egitime_hazir_veri"
    
    print("[1/5] Veriler yükleniyor...")
    X_train = np.load(os.path.join(klasor, 'X_train.npy'))
    y_train = np.load(os.path.join(klasor, 'y_train.npy'))
    X_val = np.load(os.path.join(klasor, 'X_val.npy'))
    y_val = np.load(os.path.join(klasor, 'y_val.npy'))
    
    print(f"    - Orijinal Eğitim Verisi Boyutu: {X_train.shape}")
    print(f"    - Orijinal Doğrulama Verisi Boyutu: {X_val.shape}")
    
    # MÜHENDİSLİK KARARI: Autoencoder Anomali Tespitinde sadece NORMAL (Benign) veri ile eğitilir.
    # Benign sınıfının LabelEncoder'daki numarası 0'dır. Sadece 0 olanları seçiyoruz.
    X_train_benign = X_train[y_train == 0]
    X_val_benign = X_val[y_val == 0]
    print(f"    - Sadece Normal (Benign) Eğitim Verisi Boyutu: {X_train_benign.shape}")
    print(f"    - Sadece Normal (Benign) Doğrulama Verisi Boyutu: {X_val_benign.shape}")
    
    # PyTorch Tensors'a çevirme
    X_train_tensor = torch.FloatTensor(X_train_benign)
    X_val_tensor = torch.FloatTensor(X_val_benign)
    
    # DataLoader oluşturma (Veriyi Batch'ler halinde modele vermek için)
    batch_size = 2048 # Veri büyük olduğu için batch size'ı yüksek tutuyoruz (Hızlı eğitim)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor) # Autoencoder'da Hedef yine verinin kendisidir!
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Cihaz seçimi (Ekran kartı varsa onu kullan, yoksa işlemci)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/5] Model Cihazı (Device): {device}")
    
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim=input_dim).to(device)
    
    # Kayıp Fonksiyonu ve Optimizasyon
    criterion = nn.MSELoss() # Ortalama Kare Hatası (Mean Squared Error)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ERKEN DURDURMA (EARLY STOPPING) PARAMETRELERİ
    num_epochs = 100 # Maksimum 100 epoch eğit
    patience = 5     # 5 epoch boyunca validation loss iyileşmezse eğitimi durdur
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_yolu = os.path.join(klasor, "autoencoder_model.pth")
    
    print("[3/5] Autoencoder Eğitimi ve Doğrulama Takibi Başlıyor...\n")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # --- EĞİTİM AŞAMASI ---
        model.train()
        toplam_train_loss = 0
        for data in train_loader:
            veri, _ = data
            veri = veri.to(device)
            
            # Forward pass
            cikti = model(veri)
            loss = criterion(cikti, veri) # Çıktı ile orijinal girdi arasındaki farkı ölç
            
            # Backward pass (Geri yayılım ve ağırlık güncelleme)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            toplam_train_loss += loss.item()
            
        ortalama_train_loss = toplam_train_loss / len(train_loader)
        
        # --- DOĞRULAMA (VALIDATION) AŞAMASI ---
        model.eval()
        toplam_val_loss = 0
        with torch.no_grad(): # Validation sırasında gradyan takibini kapatıyoruz (Bellek ve hız tasarrufu)
            for data in val_loader:
                veri, _ = data
                veri = veri.to(device)
                cikti = model(veri)
                loss = criterion(cikti, veri)
                toplam_val_loss += loss.item()
                
        ortalama_val_loss = toplam_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs}], Train Loss (MSE): {ortalama_train_loss:.6f} | Val Loss (MSE): {ortalama_val_loss:.6f}")
        
        # --- ERKEN DURDURMA VE EN İYİ MODEL KAYDI ---
        if ortalama_val_loss < best_val_loss:
            best_val_loss = ortalama_val_loss
            epochs_no_improve = 0
            # En iyi modeli diske kaydet
            torch.save(model.state_dict(), model_yolu)
            print(f"    => Model iyileştirildi! En iyi model kaydedildi. (Val Loss: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping tetiklendi! Son {patience} epoch'tur gelişim yok.")
                print(f"En iyi modelin ulaştığı en düşük Val Loss (MSE): {best_val_loss:.6f}")
                break
        
    gecen_sure = time.time() - start_time
    print(f"\n[4/5] Eğitim tamamlandı. Toplam Süre: {gecen_sure:.2f} saniye")
    print(f"[5/5] En iyi model dosyası: {model_yolu}")
    print("\nSonraki Adım: Validation verisi ile anomali Eşik (Threshold) değerinin belirlenmesi ve Test verisi ile performans ölçümü!")

if __name__ == "__main__":
    train_autoencoder()
