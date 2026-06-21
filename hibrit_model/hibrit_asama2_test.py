import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time


# ---------------------------------------------------------
# 1. MİMARİLERİN TEKRAR TANIMLANMASI (Ağırlıkları yüklemek için)
# ---------------------------------------------------------
class TrafficAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TrafficAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TrafficLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(TrafficLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def run_hybrid_evaluation():
    klasor = "egitime_hazir_veri"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Test verileri ve modeller {device} üzerinde yükleniyor...")

    # Sadece Test verisini yüklüyoruz (Gerçek dünya simülasyonu)
    X_test = np.load(os.path.join(klasor, 'X_test.npy'))
    y_test = np.load(os.path.join(klasor, 'y_test.npy'))

    input_dim = X_test.shape[1]
    benzersiz_siniflar = np.unique(y_test)
    num_classes = len(benzersiz_siniflar)

    # Modellerin Ayağa Kaldırılması
    autoencoder = TrafficAutoencoder(input_dim).to(device)
    autoencoder.load_state_dict(
        torch.load(os.path.join(klasor, "autoencoder_filtre_model.pth"), map_location=device, weights_only=True))
    autoencoder.eval()

    lstm = TrafficLSTM(input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes).to(device)
    lstm.load_state_dict(
        torch.load(os.path.join(klasor, "lstm_trafik_model.pth"), map_location=device, weights_only=True))
    lstm.eval()

    # Eşik değerini (Threshold) dosyadan okuma
    with open(os.path.join(klasor, "autoencoder_threshold.txt"), "r") as f:
        threshold = float(f.read().strip())
    print(f"    - Autoencoder Eşik Değeri Okundu: {threshold:.6f}")

    print("\n[2/4] 1. Aşama: Autoencoder tüm test trafiğini filtreliyor...")
    start_time = time.time()

    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # RAM şişmesini önlemek için veriyi batch'ler halinde filtreden geçiriyoruz
    batch_size = 2048
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size, shuffle=False)

    mse_hatalari = []
    with torch.no_grad():
        for batch in test_loader:
            veri = batch[0]
            reconstructed = autoencoder(veri)
            mse = torch.mean((veri - reconstructed) ** 2, dim=1)
            mse_hatalari.extend(mse.cpu().numpy())

    mse_hatalari = np.array(mse_hatalari)

    # Filtreleme Mantığı: Eşiğin altındakiler Normal (0), üstündekiler Şüpheli
    supheli_indeksler = np.where(mse_hatalari > threshold)[0]
    temiz_indeksler = np.where(mse_hatalari <= threshold)[0]

    print(f"    - Toplam Test Paketi: {len(X_test)}")
    print(f"    - Eşikten Geçen (Temiz Kabul Edilen) Paket: {len(temiz_indeksler)}")
    print(f"    - Eşiğe Takılan (Şüpheli) Paket: {len(supheli_indeksler)}")

    print("\n[3/4] 2. Aşama: Şüpheli paketler LSTM sorgusuna alınıyor...")

    # Nihai tahminler dizisini oluşturup, ilk başta herkese "0 (Benign)" diyoruz
    nihai_tahminler = np.zeros(len(X_test), dtype=int)

    if len(supheli_indeksler) > 0:
        X_supheli = X_test[supheli_indeksler]

        # LSTM veriyi 3 boyutlu ister
        X_supheli_3d = np.expand_dims(X_supheli, axis=1)
        X_supheli_tensor = torch.FloatTensor(X_supheli_3d).to(device)

        supheli_loader = DataLoader(TensorDataset(X_supheli_tensor), batch_size=batch_size, shuffle=False)

        lstm_tahminleri = []
        with torch.no_grad():
            for batch in supheli_loader:
                veri = batch[0]
                ciktilar = lstm(veri)
                _, tahminler = torch.max(ciktilar, 1)
                lstm_tahminleri.extend(tahminler.cpu().numpy())

        # LSTM'in kararlarını nihai sonuç listesindeki ilgili yerlere yazıyoruz
        nihai_tahminler[supheli_indeksler] = lstm_tahminleri

    print(f"    - Test İşlemi Tamamlandı ({time.time() - start_time:.2f} sn).")

    print("\n[4/4] HİBRİT SİSTEM (Autoencoder + LSTM) PERFORMANS RAPORU")
    print("=" * 65)

    sinif_haritasi = {}
    sinif_isimleri_yolu = os.path.join(klasor, 'sinif_isimleri.txt')
    if os.path.exists(sinif_isimleri_yolu):
        with open(sinif_isimleri_yolu, 'r', encoding='utf-8') as f:
            for line in f:
                k, v = line.strip().split(':')
                sinif_haritasi[int(k)] = v

    target_names = [sinif_haritasi.get(i, f"Sınıf {i}") for i in sorted(np.unique(y_test))]

    print("\nKARMAŞIKLIK MATRİSİ (Confusion Matrix):")
    print(confusion_matrix(y_test, nihai_tahminler))

    print("\nSINIFLANDIRMA RAPORU (Classification Report):")
    print(classification_report(y_test, nihai_tahminler, target_names=target_names))

    final_acc = accuracy_score(y_test, nihai_tahminler)
    print(f"Genel Doğruluk (Accuracy) Skoru: {final_acc:.4f}")


if __name__ == "__main__":
    run_hybrid_evaluation()