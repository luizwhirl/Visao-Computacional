import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
import os

def filtro_bilateral_numpy(imagem, k, sigma_espacial, sigma_cor):
    pad = k // 2
    
    img_float = imagem.astype(np.float32)
    img_padded = np.pad(img_float, pad, mode='reflect')
    
    janelas = sliding_window_view(img_padded, window_shape=(k, k))
    
    y, x = np.ogrid[-pad:pad+1, -pad:pad+1]
    kernel_espacial = np.exp(-(x**2 + y**2) / (2 * sigma_espacial**2))
    
    centros = img_float[:, :, np.newaxis, np.newaxis]
    diff = janelas - centros
    
    kernel_cor = np.exp(-(diff**2) / (2 * sigma_cor**2))
    pesos_finais = kernel_espacial * kernel_cor
    soma_pondr = np.sum(janelas * pesos_finais, axis=(2, 3))
    soma_pesos = np.sum(pesos_finais, axis=(2, 3))
    img_filtrada = soma_pondr / soma_pesos
    
    return np.clip(img_filtrada, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    entrada = "entradaq6.png"
    
    if not os.path.exists(entrada):
        print(f"arquivo '{entrada}' não foi encontrado")
    else:
        entrada = np.array(Image.open(entrada).convert('L'))
        
        k = 9
        sigma_s = 3.0
        sigma_c = 40.0
        
        img_limpa = filtro_bilateral_numpy(entrada, k, sigma_s, sigma_c)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(entrada, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('original')
        axes[0].axis('off')
        axes[1].imshow(img_limpa, cmap='gray', vmin=0, vmax=255)
        titulo = f'resultado (Bilateral k={k}, $\sigma_s$={sigma_s}, $\sigma_c$={sigma_c:.1f})'
        axes[1].set_title(titulo)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()