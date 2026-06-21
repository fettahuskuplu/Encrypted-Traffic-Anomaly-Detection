import pandas as pd
import numpy as np
import os
import warnings

# low_memory uyarılarını kapatmak için
warnings.filterwarnings('ignore')

def temizle_ve_filtrele(girdi_klasoru, cikti_klasoru, chunk_size=500000):
    """
    Tüm veri setini tarar, sadece Port 443 olanları filtreler,
    hatalı verileri (NaN, Inf) temizler ve temiz veriyi kaydeder.
    """
    if not os.path.exists(cikti_klasoru):
        os.makedirs(cikti_klasoru)
        print(f"[*] '{cikti_klasoru}' klasörü oluşturuldu.")

    dosyalar = [f for f in os.listdir(girdi_klasoru) if f.endswith('.csv')]
    
    toplam_satir = 0
    toplam_https_satir = 0

    for dosya in dosyalar:
        dosya_yolu = os.path.join(girdi_klasoru, dosya)
        cikti_dosya_yolu = os.path.join(cikti_klasoru, f"temiz_{dosya}")
        
        print(f"\n[>] İşlenen dosya: {dosya}")
        
        chunk_iter = pd.read_csv(dosya_yolu, chunksize=chunk_size)
        ilk_yazim_mi = True
        
        for i, chunk in enumerate(chunk_iter):
            # 1. Sütun isimlerindeki boşlukları temizle
            chunk.columns = chunk.columns.str.strip()
            
            # 2. Şifreli Trafik (HTTPS Port 443 ve SSH Port 22) Filtrelemesi
            if 'Dst Port' in chunk.columns:
                chunk = chunk[chunk['Dst Port'].isin([443, 22])]
            
            if len(chunk) == 0:
                continue
            # 3. Sonsuz (Inf) değerleri NaN'a çevir ki tek seferde 0 ile doldurabilelim
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 4. Boş (NaN) değer içeren satırları at SİLME, 0 İLE DOLDUR!
            # (Siber saldırılarda sürenin 0 olması yüzünden hızı Infinity olan satırlar saldırı izidir)
            chunk.fillna(0, inplace=True)
            
            # 5. Temizlenen bu chunk'ı yeni dosyaya ekle (append)
            # Mod 'a' (append), eğer ilk yazım ise header (başlık) ekle
            chunk.to_csv(cikti_dosya_yolu, mode='a', header=ilk_yazim_mi, index=False)
            ilk_yazim_mi = False
            
            toplam_satir += chunk_size
            toplam_https_satir += len(chunk)
            
            print(f"    - Chunk {i+1} temizlendi ve kaydedildi. (Bulunan HTTPS satır: {len(chunk)})")
            
    print(f"\n[+] Tüm dosyalar tamamlandı! Toplam filtrelenen HTTPS satır sayısı: {toplam_https_satir}")

if __name__ == "__main__":
    girdi_klasoru = "dataset"
    cikti_klasoru = "temiz_veri"
    
    print("Veri Ön İşleme (Data Preprocessing) başlatılıyor...")
    temizle_ve_filtrele(girdi_klasoru, cikti_klasoru)
