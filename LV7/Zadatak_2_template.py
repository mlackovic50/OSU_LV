import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import os

def obradi_sliku(putanja, K=8):
    try:
        if not os.path.exists(putanja):
            print(f"Datoteka {putanja} ne postoji!")
            return
            
        with Image.open(putanja) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            img_np = np.array(img) / 255.0
            
            h, w, d = img_np.shape
            if d == 4:
                img_np = img_np[:,:,:3]
                d = 3
            
            pixel_values = img_np.reshape(-1, d)
            
            kmeans = MiniBatchKMeans(n_clusters=K, 
                                   random_state=42, 
                                   n_init=3,
                                   batch_size=2048)
            kmeans.fit(pixel_values)
            
            new_colors = kmeans.cluster_centers_[kmeans.labels_]
            quantized_img = new_colors.reshape(h, w, d)
            
            plt.figure(figsize=(15,5))
            
            plt.subplot(1,2,1)
            plt.imshow(img_np)
            plt.title(f'Original\n{putanja}')
            plt.axis('off')
            
            plt.subplot(1,2,2)
            plt.imshow(quantized_img)
            plt.title(f'Kvantizirana\n{K} boja')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Greška pri obradi {putanja}: {str(e)}")

# Glavni program
if __name__ == "__main__":
    sve_slike = [f"imgs/test_{i}.jpg" for i in range(1,7)]
    
    K_vrijednosti = [32, 16, 8, 4]
    
    for K in K_vrijednosti:
        print(f"\n{'='*50}")
        print(f"REZULTATI ZA K = {K}")
        print(f"{'='*50}")
        
        for putanja in sve_slike:
            if os.path.exists(putanja):
                print(f"\nObrada: {os.path.basename(putanja)}")
                obradi_sliku(putanja, K)