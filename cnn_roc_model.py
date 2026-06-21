import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
import time


# ---------------------------------------------------------
# 1. 1D CNN MİMARİSİNİN TANIMLANMASI
# ---------------------------------------------------------
class TrafficCNN(nn.Module):
    def __init__(self, input_dim):
        super(TrafficCNN, self).__init__()

        # 1 Boyutlu Evrişim (Ağ özelliklerini yan yana tarar)
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Tam Bağlantılı Katmanlar (Olasılık Üretimi)
        flattened_size = (input_dim // 4) * 64
        self.fc_layer = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Tek bir nöron çıkışı (Saldırı olma olasılığı)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)
        return x  # Raw Logits (Sigmoid daha sonra uygulanacak)


def train_cnn_with_roc():
    klasor = "egitime_hazir_veri"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Temizlenmiş veriler {device} üzerinde yükleniyor...")

    # Verilerin Yüklenmesi
    X_train = np.load(os.path.join(klasor, 'X_train.npy'))
    y_train = np.load(os.path.join(klasor, 'y_train.npy'))
    X_val = np.load(os.path.join(klasor, 'X_val.npy'))
    y_val = np.load(os.path.join(klasor, 'y_val.npy'))
    X_test = np.load(os.path.join(klasor, 'X_test.npy'))
    y_test = np.load(os.path.join(klasor, 'y_test.npy'))

    # İkili (Binary) Sınıflandırmaya Çevirme: 0 (Benign) vs 1 (Anomali)
    y_train_bin = (y_train > 0).astype(np.float32)
    y_val_bin = (y_val > 0).astype(np.float32)
    y_test_bin = (y_test > 0).astype(np.float32)

    # CNN için Boyut Genişletme (Batch, Channels, Features)
    X_train_cnn = np.expand_dims(X_train, axis=1)
    X_val_cnn = np.expand_dims(X_val, axis=1)
    X_test_cnn = np.expand_dims(X_test, axis=1)

    # DataLoader Hazırlığı
    batch_size = 2048
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_cnn), torch.FloatTensor(y_train_bin)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_cnn), torch.FloatTensor(y_val_bin)),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_cnn), torch.FloatTensor(y_test_bin)),
                             batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = TrafficCNN(input_dim).to(device)

    # Kayıp Fonksiyonu: İkili Sınıflandırma için Logit ile BCE
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15  # CNN çok hızlı öğrenir, 15 epoch yeterlidir
    model_yolu = os.path.join(klasor, "cnn_ikili_model.pth")

    print("\n[2/4] CNN Eğitim Döngüsü Başlıyor (Sıfırdan İkili Sınıflandırma)...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        toplam_loss = 0
        for veriler, etiketler in train_loader:
            veriler, etiketler = veriler.to(device), etiketler.to(device).unsqueeze(1)

            optimizer.zero_grad()
            ciktilar = model(veriler)
            loss = criterion(ciktilar, etiketler)
            loss.backward()
            optimizer.step()
            toplam_loss += loss.item()

        print(f"    Epoch [{epoch + 1:02d}/{num_epochs}] -> Train Kaybı (Loss): {toplam_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), model_yolu)
    print(f"Eğitim Tamamlandı ({time.time() - start_time:.2f} sn).")

    # ---------------------------------------------------------
    # KRİTİK AŞAMA: ROC EĞRİSİ VE YOUDEN İNDEKSİ İLE OPTİMİZASYON
    # ---------------------------------------------------------
    print("\n[3/4] Doğrulama (Validation) Seti üzerinde ROC Analizi yapılıyor...")
    model.eval()
    val_tahmin_olasiliklari = []
    val_gercek_etiketler = []

    with torch.no_grad():
        for veriler, etiketler in val_loader:
            veriler = veriler.to(device)
            ciktilar = model(veriler)
            # Logit değerlerini 0 ile 1 arası olasılıklara çeviriyoruz
            olasiliklar = torch.sigmoid(ciktilar).cpu().numpy()
            val_tahmin_olasiliklari.extend(olasiliklar)
            val_gercek_etiketler.extend(etiketler.numpy())

    val_tahmin_olasiliklari = np.array(val_tahmin_olasiliklari).flatten()
    val_gercek_etiketler = np.array(val_gercek_etiketler)

    # Sklearn ile ROC Eğrisi Değerlerini Hesaplama
    fpr, tpr, thresholds = roc_curve(val_gercek_etiketler, val_tahmin_olasiliklari)

    # Youden İndeksi Formülü: J = Sensitivity (TPR) + Specificity (1 - FPR) - 1
    # Bu formül, sahte alarmların en düşük, saldırı yakalamanın en yüksek olduğu noktayı bulur
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    optimum_threshold = thresholds[best_idx]

    print(f"    => ROC Matematiğine Göre Bulunan Optimum Karar Eşiği: {optimum_threshold:.4f}")
    print("    (Model, bir paketin saldırı olma ihtimalini bu değerin üzerinde görürse alarm verecek)")

    # ---------------------------------------------------------
    # TEST AŞAMASI: OPTİMİZE EDİLMİŞ EŞİK İLE
    # ---------------------------------------------------------
    print("\n[4/4] Optimize edilmiş eşik ile Nihai Test Başlatılıyor...")
    test_tahmin_olasiliklari = []
    test_gercek_etiketler = []

    with torch.no_grad():
        for veriler, etiketler in test_loader:
            veriler = veriler.to(device)
            ciktilar = model(veriler)
            olasiliklar = torch.sigmoid(ciktilar).cpu().numpy()
            test_tahmin_olasiliklari.extend(olasiliklar)
            test_gercek_etiketler.extend(etiketler.numpy())

    test_tahmin_olasiliklari = np.array(test_tahmin_olasiliklari).flatten()
    test_gercek_etiketler = np.array(test_gercek_etiketler)

    # Bulduğumuz matematiksel "Altın Eşiği" tahminlere uyguluyoruz
    nihai_tahminler = (test_tahmin_olasiliklari >= optimum_threshold).astype(int)

    target_names = ["Normal (Benign)", "Anomali / Saldırı"]

    print("\n" + "=" * 55)
    print("        CNN + ROC OPTİMİZASYON PERFORMANS SONUÇLARI      ")
    print("=" * 55 + "\n")

    print("KARMAŞIKLIK MATRİSİ (Confusion Matrix):")
    print(confusion_matrix(test_gercek_etiketler, nihai_tahminler))

    print("\nSINIFLANDIRMA RAPORU (Classification Report):")
    print(classification_report(test_gercek_etiketler, nihai_tahminler, target_names=target_names))

    final_acc = accuracy_score(test_gercek_etiketler, nihai_tahminler)
    print(f"Genel Doğruluk (Accuracy) Skoru: {final_acc:.4f}")


if __name__ == "__main__":
    train_cnn_with_roc()