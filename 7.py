import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import urllib.error

# --- Nova Função para ler PPM direto de uma URL ---
def read_ppm_from_url(url):
    try:
        # Adicionamos um cabeçalho simples para evitar bloqueios de alguns servidores
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as f:
            def read_word():
                word = b''
                while True:
                    c = f.read(1)
                    if not c:
                        break
                    if c == b'#':
                        f.readline()
                        continue
                    if c.isspace():
                        if word:
                            return word
                        continue
                    word += c
                return word

            magic = read_word().decode('ascii')
            width = int(read_word())
            height = int(read_word())
            maxval = int(read_word())

            if magic == 'P3':
                # Lê o resto dos dados e converte
                data = f.read().split()
                img = np.array(data, dtype=np.uint8).reshape((height, width, 3))
                return img
            else:
                raise ValueError(f"Formato não suportado: {magic}. Este código suporta apenas P3 (ASCII).")
                
    except urllib.error.URLError as e:
        print(f"Erro ao acessar a URL: {e}")
        return None

# --- Mantida a mesma função de filtro ---
def apply_filter(image, n, filter_matrix, save_path=None, return_float=False):        
    pad = n // 2
    kernel_flipped = np.flip(filter_matrix)
    img_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    h, w, c = image.shape
    output = np.zeros((h, w, c), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            output += img_padded[i:i+h, j:j+w, :] * kernel_flipped[i, j]
            
    if return_float:
        return output 
    else:
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output

# --- Execução Principal ---
if __name__ == "__main__":
    # 1. Carregar a imagem da internet (Link alternativo do Mandrill)
    url_imagem = "https://people.sc.fsu.edu/~jburkardt/data/ppma/blackbuck.ascii.ppm"
    
    print("Baixando e processando a imagem da URL... Isso pode levar alguns segundos.")
    imagem_rgb = read_ppm_from_url(url_imagem)
    
    if imagem_rgb is None:
        exit()

    # 2. Converter para Tons de Cinza 
    imagem_gray = np.dot(imagem_rgb[...,:3], [0.2989, 0.5870, 0.1140])
    imagem_gray_3d = imagem_gray.reshape((imagem_gray.shape[0], imagem_gray.shape[1], 1))

    # 3. Definir os Kernels de Sobel (3x3)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # 4. Aplicar os filtros 
    n = 3
    gx_float = apply_filter(imagem_gray_3d, n, sobel_x, return_float=True)
    gy_float = apply_filter(imagem_gray_3d, n, sobel_y, return_float=True)

    # 5. Calcular a Magnitude do Gradiente
    magnitude = np.sqrt(gx_float**2 + gy_float**2)

    # 6. Preparar imagens para exibição
    gx_view = np.clip(np.abs(gx_float), 0, 255).astype(np.uint8).squeeze()
    gy_view = np.clip(np.abs(gy_float), 0, 255).astype(np.uint8).squeeze()
    magnitude_view = np.clip(magnitude, 0, 255).astype(np.uint8).squeeze()
    original_view = imagem_gray.astype(np.uint8)

    # 7. Plotar os resultados
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original_view, cmap='gray')
    axes[0, 0].set_title('Imagem Original (Tons de Cinza)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gx_view, cmap='gray')
    axes[0, 1].set_title('Arestas Verticais (Gx)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gy_view, cmap='gray')
    axes[1, 0].set_title('Arestas Horizontais (Gy)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(magnitude_view, cmap='gray')
    axes[1, 1].set_title('Magnitude do Gradiente')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()