import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time


# 1. LSTM MİMARİSİNİN TANIMLANMASI
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


def train_and_evaluate_lstm():
    klasor = "egitime_hazir_veri"

    print("[1/5] Hazırlanan Numpy verileri yükleniyor...")
    X_train = np.load(os.path.join(klasor, 'X_train.npy'))
    y_train = np.load(os.path.join(klasor, 'y_train.npy'))
    X_val = np.load(os.path.join(klasor, 'X_val.npy'))
    y_val = np.load(os.path.join(klasor, 'y_val.npy'))
    X_test = np.load(os.path.join(klasor, 'X_test.npy'))
    y_test = np.load(os.path.join(klasor, 'y_test.npy'))

    benzersiz_siniflar = np.unique(y_train)
    num_classes = len(benzersiz_siniflar)
    print(f"    - Tespit edilen benzersiz sınıf sayısı: {num_classes} {benzersiz_siniflar}")

    X_train_3d = np.expand_dims(X_train, axis=1)
    X_val_3d = np.expand_dims(X_val, axis=1)
    X_test_3d = np.expand_dims(X_test, axis=1)

    X_train_tensor = torch.FloatTensor(X_train_3d)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_3d)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_3d)
    y_test_tensor = torch.LongTensor(y_test)

    batch_size = 2048
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/5] Model donanım hızlandırıcı cihazı (Device): {device}")

    input_dim = X_train.shape[1]
    hidden_dim = 64
    num_layers = 2

    model = TrafficLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)

    # =========================================================================
    # YENİ BÖLÜM: MANUEL VE DENGELİ SINIF AĞIRLIKLANDIRMASI (SWEET SPOT)
    # =========================================================================
    # Sklearn'ün otomatik 94 katlık ağır cezası yerine, optimum denge için
    # Normal trafiğe 1.0, Saldırı trafiğine 15.0 ceza katsayısı veriyoruz.
    manuel_agirliklar = [1.0, 15.0]
    sinif_agirliklari_tensor = torch.FloatTensor(manuel_agirliklar).to(device)
    print(f"    - Manuel Optimize Edilmiş Ceza Çarpanları: {manuel_agirliklar}")

    # Kayıp fonksiyonu artık bu dengeli ceza çarpanlarını dikkate alacak
    criterion = nn.CrossEntropyLoss(weight=sinif_agirliklari_tensor)
    # =========================================================================

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_yolu = os.path.join(klasor, "lstm_trafik_model.pth")

    print("[3/5] LSTM Eğitim ve Doğrulama Döngüsü Başlatılıyor...\n")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        toplam_train_loss = 0
        dogru_tahmin_train = 0
        toplam_ornek_train = 0

        for veriler, etiketler in train_loader:
            veriler, etiketler = veriler.to(device), etiketler.to(device)

            optimizer.zero_grad()
            ciktilar = model(veriler)
            loss = criterion(ciktilar, etiketler)
            loss.backward()
            optimizer.step()

            toplam_train_loss += loss.item()
            _, tahminler = torch.max(ciktilar, 1)
            dogru_tahmin_train += (tahminler == etiketler).sum().item()
            toplam_ornek_train += etiketler.size(0)

        ortalama_train_loss = toplam_train_loss / len(train_loader)
        train_acc = dogru_tahmin_train / toplam_ornek_train

        model.eval()
        toplam_val_loss = 0
        dogru_tahmin_val = 0
        toplam_ornek_val = 0

        with torch.no_grad():
            for veriler, etiketler in val_loader:
                veriler, etiketler = veriler.to(device), etiketler.to(device)
                ciktilar = model(veriler)
                loss = criterion(ciktilar, etiketler)

                toplam_val_loss += loss.item()
                _, tahminler = torch.max(ciktilar, 1)
                dogru_tahmin_val += (tahminler == etiketler).sum().item()
                toplam_ornek_val += etiketler.size(0)

        ortalama_val_loss = toplam_val_loss / len(val_loader)
        val_acc = dogru_tahmin_val / toplam_ornek_val

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] -> Train Loss: {ortalama_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {ortalama_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if ortalama_val_loss < best_val_loss:
            best_val_loss = ortalama_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_yolu)
            print(f"    => Gelişme kaydedildi! Model diske yazıldı.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping tetiklendi. Son {patience} epoch'tur gelişim yok.")
                break

    gecen_sure = time.time() - start_time
    print(f"\n[4/5] Eğitim Tamamlandı. Toplam Süre: {gecen_sure:.2f} saniye.")

    print("\n[5/5] En iyi model yükleniyor ve nihai test süreci başlatılıyor...")
    model.load_state_dict(torch.load(model_yolu))
    model.eval()

    tum_tahminler = []
    tum_gercekler = []

    with torch.no_grad():
        for veriler, etiketler in test_loader:
            veriler = veriler.to(device)
            ciktilar = model(veriler)
            _, tahminler = torch.max(ciktilar, 1)

            tum_tahminler.extend(tahminler.cpu().numpy())
            tum_gercekler.extend(etiketler.numpy())

    tum_tahminler = np.array(tum_tahminler)
    tum_gercekler = np.array(tum_gercekler)

    sinif_haritasi = {}
    sinif_isimleri_yolu = os.path.join(klasor, 'sinif_isimleri.txt')
    if os.path.exists(sinif_isimleri_yolu):
        with open(sinif_isimleri_yolu, 'r', encoding='utf-8') as f:
            for line in f:
                k, v = line.strip().split(':')
                sinif_haritasi[int(k)] = v

    target_names = [sinif_haritasi.get(i, f"Sınıf {i}") for i in sorted(np.unique(tum_gercekler))]

    print("\n" + "=" * 55)
    print("             LSTM MODEL PERFORMANS SONUÇLARI             ")
    print("=" * 55 + "\n")

    print("KARMAŞIKLIK MATRİSİ (Confusion Matrix):")
    print(confusion_matrix(tum_gercekler, tum_tahminler))

    print("\nSINIFLANDIRMA RAPORU (Classification Report):")
    print(classification_report(tum_gercekler, tum_tahminler, target_names=target_names))

    final_acc = accuracy_score(tum_gercekler, tum_tahminler)
    print(f"Genel Doğruluk (Accuracy) Skoru: {final_acc:.4f}")


if __name__ == "__main__":
    train_and_evaluate_lstm()