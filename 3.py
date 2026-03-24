# 1. Baixando a imagem de teste
# !wget -q https://people.sc.fsu.edu/~jburkardt/data/ppma/snail.ascii.ppm -O caracol.ppm

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# FUNÇÕES DAS QUESTÕES 1 E 2
# ==========================================
def read_ppm(filepath):
    """Lê arquivo PPM."""
    with open(filepath, 'rb') as f:
        def read_word():
            word = b''
            while True:
                c = f.read(1)
                if not c: break
                if c == b'#':
                    f.readline(); continue
                if c.isspace():
                    if word: return word
                    continue
                word += c
            return word

        magic = read_word().decode('ascii')
        width = int(read_word())
        height = int(read_word())
        maxval = int(read_word())

        if magic == 'P6':
            data = f.read()
            expected_size = width * height * 3
            img = np.frombuffer(data[:expected_size], dtype=np.uint8).reshape((height, width, 3))
        elif magic == 'P3':
            data = f.read().split()
            img = np.array(data, dtype=np.uint8).reshape((height, width, 3))
        else:
            raise ValueError("Formato não suportado.")
        return img

def apply_filter(image, n, filter_matrix):
    """Aplica o filtro espacial n x n à imagem (Suporta Tons de Cinza 2D e RGB 3D)."""
    pad = n // 2
    kernel_flipped = np.flip(filter_matrix)
    
    # Verifica se a imagem é 2D (tons de cinza) ou 3D (colorida)
    is_grayscale = (len(image.shape) == 2)
    
    if is_grayscale:
        img_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
        h, w = image.shape
        output = np.zeros((h, w), dtype=np.float32)
        
        for i in range(n):
            for j in range(n):
                output += img_padded[i:i+h, j:j+w] * kernel_flipped[i, j]
    else:
        img_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        h, w, c = image.shape
        output = np.zeros((h, w, c), dtype=np.float32)
        
        for i in range(n):
            for j in range(n):
                output += img_padded[i:i+h, j:j+w, :] * kernel_flipped[i, j]
                
    return np.clip(output, 0, 255).astype(np.uint8)

def criar_filtro_gaussiano(k, sigma):
    """Gera a matriz do filtro Gaussiano normalizado."""
    limite = (k - 1) // 2
    vetor_coord = np.arange(-limite, limite + 1)
    x, y = np.meshgrid(vetor_coord, vetor_coord)
    
    termo_constante = 1 / (2 * np.pi * sigma**2)
    expoente = -(x**2 + y**2) / (2 * sigma**2)
    
    H = termo_constante * np.exp(expoente)
    return H / np.sum(H) 

# ==========================================
# EXECUÇÃO DA QUESTÃO 3
# ==========================================
if __name__ == "__main__":
    # 1. Carregar a imagem RGB e converter para Tons de Cinza
    imagem_rgb = read_ppm("caracol.ppm")
    # Fórmula da luminosidade: Y = 0.299*R + 0.587*G + 0.114*B
    imagem_cinza = np.dot(imagem_rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    
    # 2. Definir 3 variações de k e 3 variações de sigma (total 9 combinações)
    # A dica sugere sigma ≈ k/6. Usaremos k = 5, 15, 25.
    valores_k = [5, 15, 25]
    valores_sigma = [1.0, 2.5, 4.0] 
    
    # 3. Preparar a grade de exibição (Plot)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Suavização Gaussiana: 9 Combinações de $k$ e $\sigma$', fontsize=16)
    
    print("Processando as 9 combinações. Aguarde um instante...")
    
    # 4. Loop para testar todas as combinações
    for i, k in enumerate(valores_k):
        for j, sigma in enumerate(valores_sigma):
            # Gera o filtro
            filtro = criar_filtro_gaussiano(k, sigma)
            
            # Aplica a convolução
            img_suavizada = apply_filter(imagem_cinza, k, filtro)
            
            # Plota na posição correta da grade
            ax = axes[i, j]
            ax.imshow(img_suavizada, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'k={k}, $\sigma$={sigma}')
            ax.axis('off')
            
    plt.tight_layout()
    # Ajuste para o título principal não sobrepor os gráficos
    fig.subplots_adjust(top=0.92) 
    plt.show()