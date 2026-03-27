import cv2
import matplotlib.pyplot as plt

# 1. Carregar a imagem solicitada
caminho_imagem = 'entradaq6.png'
img = cv2.imread(caminho_imagem)

if img is None:
    print(f"Erro: Não foi possível carregar a imagem '{caminho_imagem}'.")
else:
    # Convertendo de BGR para RGB para exibir corretamente
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # 2. Abordagem 1: Aumentar a janela do Filtro de Mediana
    # Usando janela 7x7 (vai remover mais ruído, mas pode borrar mais)
    img_mediana_7 = cv2.medianBlur(img_rgb, 7)

    # 3. Abordagem 2: Filtro Non-Local Means Denoising (O "Estado da Arte" clássico)
    # Este filtro é não-linear e excelente para preservar bordas.
    # Parâmetros: imagem, dst, h (força do filtro para luminância), hColor (força para cor), templateWindowSize, searchWindowSize
    # Pode demorar alguns segundinhos a mais para processar!
    img_nlm = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)

    # 4. Plotar os resultados lado a lado
    plt.figure(figsize=(18, 6))

    # Plot Imagem Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('1. Imagem Original')
    plt.axis('off')

    # Plot Mediana 7x7
    plt.subplot(1, 3, 2)
    plt.imshow(img_mediana_7)
    plt.title('2. Filtro Mediana (7x7)')
    plt.axis('off')

    # Plot Non-Local Means
    plt.subplot(1, 3, 3)
    plt.imshow(img_nlm)
    plt.title('3. Non-Local Means (Avançado)')
    plt.axis('off')

    # Exibir na tela
    plt.tight_layout()
    plt.show()