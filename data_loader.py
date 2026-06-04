import pandas as pd
import os

def process_first_chunk(file_path, chunk_size=100000):
    """
    Belirtilen CSV dosyasından veriyi chunk'lar (parçalar) halinde okur
    ve sadece HTTPS trafiğini (Hedef Port 443) filtreler.
    
    Argümanlar:
        file_path: Okunacak CSV dosyasının yolu.
        chunk_size: Tek seferde belleğe alınacak satır sayısı.
    """
    print(f"[{file_path}] dosyası okunuyor, hedef port 443 filtrelemesi uygulanıyor...")
    
    # Chunk boyutunu belirliyoruz. Veri seti çok büyük olduğu için hepsini
    # belleğe (RAM) almak yerine parça parça okuyacağız.
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    
    # Sadece ilk chunk üzerinde işlem yapıp sonucunu görelim
    for i, chunk in enumerate(chunk_iter):
        # Kolon isimlerinde başta/sonda gereksiz boşluklar olabiliyor (Örn: ' Dst Port ')
        # Bunları temizleyelim ki filtreleme yaparken hata almayalım.
        chunk.columns = chunk.columns.str.strip()
        
        # Sadece Destination Port (Dst Port) 443 olan satırları alıyoruz.
        # Bu sayede sadece şifrelenmiş (HTTPS) trafiği izole ediyoruz.
        https_traffic = chunk[chunk['Dst Port'] == 443]
        
        print(f"Chunk {i+1} okundu. Toplam satır: {len(chunk)}, HTTPS (Port 443) satır sayısı: {len(https_traffic)}")
        
        if len(https_traffic) > 0:
            print("İlk HTTPS veri örneği:")
            print(https_traffic.head())
            break # HTTPS trafiği bulduğumuzda ilk örneği gösterip çıkalım.
            
        if i >= 4:
            print("İlk 5 chunk içinde HTTPS trafiğine rastlanmadı.")
            break

if __name__ == "__main__":
    # Proje dizinindeki dataset klasörünün içindeki ilk dosyayı seçelim
    dataset_dir = "dataset"
    first_file = "02-14-2018.csv"
    file_path = os.path.join(dataset_dir, first_file)
    
    if os.path.exists(file_path):
        process_first_chunk(file_path, chunk_size=100000)
    else:
        print(f"Hata: {file_path} dosyası bulunamadı. Lütfen dataset klasörünün doğru yerde olduğundan emin ol.")
