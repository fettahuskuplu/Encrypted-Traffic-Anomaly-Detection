import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def ozellik_muhendisligi_yap(girdi_klasoru, cikti_klasoru):
    if not os.path.exists(cikti_klasoru):
        os.makedirs(cikti_klasoru)
        print(f"[*] '{cikti_klasoru}' klasörü oluşturuldu.")

    dosyalar = [f for f in os.listdir(girdi_klasoru) if f.endswith('.csv')]
    
    print("[1/6] Temizlenen veri dosyaları belleğe (RAM) yükleniyor. Bu biraz zaman alabilir...")
    df_listesi = []
    for dosya in dosyalar:
        dosya_yolu = os.path.join(girdi_klasoru, dosya)
        # Sütun tiplerini daha az yer kaplaması için float32'ye indirebiliriz ama şimdilik standart okuyalım.
        df_listesi.append(pd.read_csv(dosya_yolu, low_memory=False))
        
    # Tüm dosyaları tek bir büyük tabloda (DataFrame) birleştirelim
    tam_veri = pd.concat(df_listesi, ignore_index=True)
    print(f"[+] Veri başarıyla birleştirildi. Toplam Satır: {len(tam_veri)}, Sütun: {tam_veri.shape[1]}")

    print("[2/6] Gereksiz ve metin (String) içeren özellikler çıkartılıyor...")
    # CSE-CIC-IDS2018 veri setinin bazı günlerinde 'Flow ID', 'Src IP', 'Dst IP' gibi string kolonlar bulunur.
    # Bunlar hem ezbere (overfitting) yol açar hem de MinMaxScaler'ı kilitler. Bu yüzden sayısal olmayan her şeyi atıyoruz.
    sayisal_olmayan_kolonlar = tam_veri.select_dtypes(include=['object']).columns.tolist()
    
    # 'Label' bizim hedefimiz, onu silmemeliyiz. Listeden çıkarıyoruz.
    if 'Label' in sayisal_olmayan_kolonlar:
        sayisal_olmayan_kolonlar.remove('Label')
        
    # Geriye kalan tüm string/gereksiz kolonları siliyoruz (Örn: Timestamp, Src IP, Dst IP, Flow ID)
    if sayisal_olmayan_kolonlar:
        tam_veri.drop(columns=sayisal_olmayan_kolonlar, inplace=True)
        print(f"    - Silinen Metin Kolonları: {sayisal_olmayan_kolonlar}")
        
    # MÜHENDİSLİK KARARI: Modellerin port numarasını (22 veya 443) veya protokol değerini ezberlemesini önlemek için
    # 'Dst Port' ve 'Protocol' sütunlarını manuel olarak kaldırıyoruz.
    zorunlu_silinecekler = ['Dst Port', 'Protocol']
    for col in zorunlu_silinecekler:
        if col in tam_veri.columns:
            tam_veri.drop(columns=[col], inplace=True)
            print(f"    - Silinen Güvenlik Sızıntısı Kolonu (Port/Protocol): {col}")
        
    print("[3/6] Hedef değişken (Label) ve Özellikler (X) ayrılıyor ve Gizli Hatalar Temizleniyor...")
    # 'Label' hariç tüm kolonları bulalım ve zorla sayısala çevirelim. 
    # 'Infinity' veya '-' gibi stringler bu sayede zorla NaN (Boş) değerine dönüşecek.
    ozellik_kolonlari = [col for col in tam_veri.columns if col != 'Label']
    tam_veri[ozellik_kolonlari] = tam_veri[ozellik_kolonlari].apply(pd.to_numeric, errors='coerce')
    
    # Yeni oluşan NaN değerleri SİLMEK YERİNE 0 İLE DOLDURALIM
    eski_satir_sayisi = len(tam_veri)
    tam_veri.fillna(0, inplace=True)
    yeni_satir_sayisi = len(tam_veri)
    print(f"    - Gizli/Bozuk metinler tespit edildi ve 0 ile doldurularak SALDIRI verileri kurtarıldı!")
    
    # MÜHENDİSLİK ADIMI: Varyansı çok düşük (<1e-5) veya tamamen 0 (sabit) olan özellikleri eliyoruz.
    # Bu, gürültüyü azaltır ve modelin normal ile anomali trafiği ayırt etmesine yardımcı olur.
    ozellik_kolonlari = [col for col in tam_veri.columns if col != 'Label']
    varyanslar = tam_veri[ozellik_kolonlari].var()
    dusuk_varyansli_kolonlar = varyanslar[varyanslar < 1e-5].index.tolist()
    
    if dusuk_varyansli_kolonlar:
        tam_veri.drop(columns=dusuk_varyansli_kolonlar, inplace=True)
        print(f"    - Düşük Varyanslı (<1e-5) / Sabit {len(dusuk_varyansli_kolonlar)} adet kolon çıkarıldı.")
        
        # Hangi kolonların neden çıkarıldığını kaydetmek için dosya oluşturuyoruz:
        with open(os.path.join(cikti_klasoru, 'cikarilan_kolonlar.txt'), 'w', encoding='utf-8') as f:
            f.write("ÇIKARILAN DÜŞÜK VARYANSLI VEYA SABİT KOLONLAR\n")
            f.write("=============================================\n\n")
            for col in dusuk_varyansli_kolonlar:
                f.write(f"- {col} (Varyans: {varyanslar[col]:.8f})\n")
    
    # Şimdi güvenle ayrım yapabiliriz
    y = tam_veri['Label'].values
    X = tam_veri.drop(columns=['Label']).values

    print("[4/6] Etiketler numaralara dönüştürülüyor (İkili / Binary Encoding)...")
    # 'Benign' -> 0, Diğer tüm saldırılar (Infilteration, SSH-BruteForce vb.) -> 1
    y_encoded = np.where(y == 'Benign', 0, 1)
    siniflar = np.array(['Benign', 'Anomaly'])
    print("    - Sınıf Dağılımı Sırasi (0: Normal, 1: Anomali):", siniflar)
    # Sınıf miktarlarını yazdıralım
    benzersiz, sayilar = np.unique(y_encoded, return_counts=True)
    for b, s in zip(benzersiz, sayilar):
        label_name = "Benign (Normal)" if b == 0 else "Anomaly (Saldırı)"
        print(f"      {label_name}: {s}")

    print("[5/6] Veri Eğitim (%70), Doğrulama (%15) ve Test (%15) olarak bölünüyor...")
    # Çift aşamalı split ile %70 - %15 - %15 oranlarını elde ediyoruz ve sınıf dengesini (stratify) koruyoruz.
    # İlk split: %70 Eğitim ve %30 Geçici Küme (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
    )
    # İkinci split: %30'luk geçici kümeyi yarı yarıya bölerek %15 Doğrulama ve %15 Test elde ediyoruz.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # --- YENİ: Sadece Eğitim Setini Dengeleme (Under-sampling) ---
    print("    - Eğitim setindeki sınıf dengesizliği gideriliyor (Under-sampling)...")
    idx_class_0 = np.where(y_train == 0)[0]
    idx_class_1 = np.where(y_train == 1)[0]
    
    num_class_1 = len(idx_class_1)
    
    # 0 sınıfından 1 sınıfının sayısı kadar rastgele örnek seçelim
    np.random.seed(42)
    sampled_idx_class_0 = np.random.choice(idx_class_0, size=num_class_1, replace=False)
    
    # İndeksleri birleştirip karıştıralım
    balanced_train_idx = np.concatenate([sampled_idx_class_0, idx_class_1])
    np.random.shuffle(balanced_train_idx)
    
    X_train = X_train[balanced_train_idx]
    y_train = y_train[balanced_train_idx]
    print(f"    - Eğitim seti dengelendi! Yeni X_train boyutu: {X_train.shape}")
    print(f"      Sınıf 0 (Benign): {np.sum(y_train == 0)}, Sınıf 1 (Anomaly): {np.sum(y_train == 1)}")


    print("[6/6] Özellikler 0 ile 1 arasına ölçeklendiriliyor (Min-Max Scaling)...")
    scaler = MinMaxScaler()
    # Sadece eğitim verisinden öğren (fit), üç kümeye de uygula (transform).
    # Bu, doğrulama ve test verilerinden bilgi sızmasını (Data Leakage) önler.
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("[+] İşlemler tamamlandı, NPY (Numpy Array) dosyaları kaydediliyor...")
    np.save(os.path.join(cikti_klasoru, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(cikti_klasoru, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(cikti_klasoru, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(cikti_klasoru, 'y_train.npy'), y_train)
    np.save(os.path.join(cikti_klasoru, 'y_val.npy'), y_val)
    np.save(os.path.join(cikti_klasoru, 'y_test.npy'), y_test)
    
    # Sınıf isimlerini de metin olarak kaydedelim ki daha sonra Confusion Matrix çizerken lazım olacak
    with open(os.path.join(cikti_klasoru, 'sinif_isimleri.txt'), 'w', encoding='utf-8') as f:
        for i, sinif in enumerate(siniflar):
            f.write(f"{i}:{sinif}\n")

    print(f"\n[!!!] BAŞARILI! Eğitime hazır veriler '{cikti_klasoru}' klasörüne kaydedildi.")

if __name__ == "__main__":
    girdi_klasoru = "temiz_veri"
    cikti_klasoru = "egitime_hazir_veri"
    ozellik_muhendisligi_yap(girdi_klasoru, cikti_klasoru)
