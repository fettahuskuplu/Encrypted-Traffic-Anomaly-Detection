import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import time

# Autoencoder mimarisini tekrar tanımlamamız gerekiyor (Modeli yüklemek için gerekli)
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def test_ve_anomali_tespiti():
    klasor = "egitime_hazir_veri"
    
    print("[1/4] Veriler (Doğrulama ve Test kümeleri) yükleniyor...")
    X_val = np.load(os.path.join(klasor, 'X_val.npy'))
    y_val = np.load(os.path.join(klasor, 'y_val.npy'))
    X_test = np.load(os.path.join(klasor, 'X_test.npy'))
    y_test = np.load(os.path.join(klasor, 'y_test.npy'))
    
    input_dim = X_test.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[2/4] Eğitilmiş Autoencoder Modeli Yükleniyor...")
    model = Autoencoder(input_dim=input_dim).to(device)
    model_yolu = os.path.join(klasor, "autoencoder_model.pth")
    model.load_state_dict(torch.load(model_yolu, map_location=device))
    model.eval() # Modeli test moduna alıyoruz (Ağırlıklar güncellenmeyecek)

    print("[3/4] Eşik Değeri (Threshold) Doğrulama (Validation) verisi ile belirleniyor...")
    # Doğrulama verisini tensör yapıp modele gönderelim
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        X_val_tahmin = model(X_val_tensor)
        val_sure = time.time() - start_time
    
    X_val_tahmin_np = X_val_tahmin.cpu().numpy()
    
    # Doğrulama verisi için Yeniden İnşa Hatası (MSE) hesaplayalım
    val_hatalar = np.mean(np.power(X_val - X_val_tahmin_np, 2), axis=1)
    
    # Sadece normal doğrulama verisinin hatalarını seçelim
    normal_val_hatalar = val_hatalar[y_val == 0]
    
    # EŞİK DEĞERİ KURALI: Normal doğrulama verilerinin hatalarının %99'unu eşik değeri kabul et.
    # Yani normal doğrulama verisinin sadece %1'ini yanlışlıkla anomali saysın.
    esik_degeri = np.percentile(normal_val_hatalar, 99)
    print(f"    - {len(X_val)} adet doğrulama paketi tarandı ({val_sure:.2f} saniye).")
    print(f"    - [!] BELİRLENEN EŞİK DEĞERİ (THRESHOLD): {esik_degeri:.6f}")
    
    print("\n[4/4] Nihai Model Testi başlatılıyor ve Anomali Kararı veriliyor...")
    # Test verisini tensör yapıp modele gönderelim
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        X_test_tahmin = model(X_test_tensor)
        test_sure = time.time() - start_time
        
    X_test_tahmin_np = X_test_tahmin.cpu().numpy()
    
    # Test verisi için Yeniden İnşa Hatası (MSE) hesaplayalım
    hatalar = np.mean(np.power(X_test - X_test_tahmin_np, 2), axis=1)
    print(f"    - {len(X_test)} adet ağ paketi sadece {test_sure:.2f} saniyede tarandı ve test edildi!")
    
    # Eğer hata belirlenen esik_degerinden büyükse 1 (Saldırı/Anomali), küçükse 0 (Normal) de
    tahmin_edilen_etiketler = (hatalar > esik_degeri).astype(int)
    
    print("\n=======================================================")
    print("                 PERFORMANS SONUÇLARI                  ")
    print("=======================================================\n")
    
    print("KARMAŞIKLIK MATRİSİ (Confusion Matrix):")
    cm = confusion_matrix(y_test, tahmin_edilen_etiketler)
    print(f"Gerçek Normal (Benign) ve Doğru Tahmin (TN): {cm[0][0]}")
    print(f"Gerçek Normal ama Yanlışlıkla Saldırı Dediği (FP): {cm[0][1]}")
    if cm.shape == (2,2):
        print(f"Gerçek Saldırı ama Gözden Kaçan (FN): {cm[1][0]}")
        print(f"Gerçek Saldırı ve Yakalanan (TP): {cm[1][1]}")
    else:
        print("Dikkat: Test verisinde hiç saldırı paketi bulunmuyor olabilir.")
        
    print("\nSINIFLANDIRMA RAPORU (Classification Report):")
    print(classification_report(y_test, tahmin_edilen_etiketler, target_names=["Normal (0)", "Saldırı (1)"] if len(np.unique(y_test)) > 1 else ["Normal (0)"]))
    
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, hatalar)
        print(f"\nROC-AUC Skoru: {roc_auc:.4f} (1.0'a ne kadar yakınsa model o kadar mükemmeldir)")

if __name__ == "__main__":
    test_ve_anomali_tespiti()
