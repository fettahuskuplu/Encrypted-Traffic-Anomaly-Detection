import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


# 1. AUTOENCODER MİMARİSİNİN TANIMLANMASI
class TrafficAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TrafficAutoencoder, self).__init__()

        # Sıkıştırma (Encoder) Katmanı
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Darboğaz (Bottleneck) - Verinin özü
        )

        # Geri Çözme (Decoder) Katmanı
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)  # Orijinal boyuta dönüş
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder():
    klasor = "egitime_hazir_veri"

    print("[1/4] Temizlenmiş veriler yükleniyor...")
    X_train = np.load(os.path.join(klasor, 'X_train.npy'))
    y_train = np.load(os.path.join(klasor, 'y_train.npy'))
    X_val = np.load(os.path.join(klasor, 'X_val.npy'))
    y_val = np.load(os.path.join(klasor, 'y_val.npy'))

    # MÜHENDİSLİK ADIMI: Dinamik İkili (Binary) Dönüşüm
    # 0 = Benign (Normal), 0'dan büyük her şey = 1 (Anomali)
    y_train_bin = (y_train > 0).astype(int)
    y_val_bin = (y_val > 0).astype(int)

    # Autoencoder SADECE normal (0) trafik ile eğitilir
    normal_indeksler = np.where(y_train_bin == 0)[0]
    X_train_normal = X_train[normal_indeksler]

    print(f"    - Toplam Eğitim Verisi: {len(X_train)}")
    print(f"    - Sadece Normal Veri (Autoencoder için): {len(X_train_normal)}")

    # PyTorch Tensörleri ve DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    - Donanım: {device}")

    X_train_tensor = torch.FloatTensor(X_train_normal).to(device)
    train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=2048, shuffle=True)

    # Model Kurulumu
    input_dim = X_train.shape[1]
    model = TrafficAutoencoder(input_dim).to(device)

    # Kayıp Fonksiyonu (MSE - Yeniden İnşa Hatası)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    model_yolu = os.path.join(klasor, "autoencoder_filtre_model.pth")

    print("\n[2/4] Autoencoder Eğitim Döngüsü Başlıyor...")
    start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
        toplam_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            toplam_loss += loss.item()

        ortalama_loss = toplam_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch [{epoch + 1:02d}/{num_epochs}] -> Yeniden İnşa Hatası (MSE Loss): {ortalama_loss:.6f}")

    torch.save(model.state_dict(), model_yolu)
    print(f"\n[3/4] Eğitim Tamamlandı ({time.time() - start_time:.2f} sn). Model diske kaydedildi.")

    # ---------------------------------------------------------
    # KRİTİK AŞAMA: ANOMALİ EŞİK DEĞERİ (THRESHOLD) HESAPLAMA
    # ---------------------------------------------------------
    print("\n[4/4] 2. Aşama için Anomali Eşik Değeri (Threshold) hesaplanıyor...")

    # Doğrulama setindeki normal verileri alalım
    val_normal_indeksler = np.where(y_val_bin == 0)[0]
    X_val_normal = X_val[val_normal_indeksler]
    X_val_normal_tensor = torch.FloatTensor(X_val_normal).to(device)

    model.eval()
    with torch.no_grad():
        reconstructed_val = model(X_val_normal_tensor)
        # Her bir paketin yeniden inşa hatasını ayrı ayrı hesaplıyoruz
        mse_per_sample = torch.mean((X_val_normal_tensor - reconstructed_val) ** 2, dim=1).cpu().numpy()

    # Normal verilerin %95'inin altında kaldığı hata değerini eşik olarak belirliyoruz
    threshold = np.percentile(mse_per_sample, 95)

    # Eşik değerini 2. aşama (LSTM) kullanabilsin diye diske yazıyoruz
    threshold_yolu = os.path.join(klasor, "autoencoder_threshold.txt")
    with open(threshold_yolu, "w") as f:
        f.write(str(threshold))

    print(f"    => Hesaplanan %95 Güvenlik Eşik Değeri: {threshold:.6f}")
    print("    => Bu değerin üzerindeki hata veren her paket 2. Aşamada (LSTM/CNN) sorguya çekilecek.")
    print("\n>>> 1. AŞAMA (FİLTRELEME) BAŞARIYLA TAMAMLANDI! <<<")


if __name__ == "__main__":
    train_autoencoder()