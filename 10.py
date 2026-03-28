import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

# 1. Carregar a imagem da URL fornecida
url = 'https://www.wondercide.com/cdn/shop/articles/Upside_down_gray_cat.png?v=1685551065'
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
img = cv2.imdecode(image, cv2.IMREAD_COLOR)

# Converter de BGR (padrão OpenCV) para RGB (para exibir corretamente no Matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Obter as dimensões originais e calcular as novas dimensões (Quadruplicar)
altura_orig, largura_orig = img_rgb.shape[:2]
nova_largura = largura_orig * 4
nova_altura = altura_orig * 4

# 3. Aplicar o redimensionamento com Interpolação Linear (Bilinear)
# cv2.INTER_LINEAR é o padrão do OpenCV para interpolação linear em 2D
img_linear = cv2.resize(img_rgb, (nova_largura, nova_altura), interpolation=cv2.INTER_LINEAR)

# 4. Aplicar o redimensionamento com Interpolação Bicúbica
img_bicubica = cv2.resize(img_rgb, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)

# 5. Exibir os resultados lado a lado
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img_rgb)
ax1.set_title(f"Original\n({largura_orig}x{altura_orig})")
ax1.axis('off')

ax2.imshow(img_linear)
ax2.set_title(f"Interpolação Linear\n({nova_largura}x{nova_altura})")
ax2.axis('off')

ax3.imshow(img_bicubica)
ax3.set_title(f"Interpolação Bicúbica\n({nova_largura}x{nova_altura})")
ax3.axis('off')

plt.tight_layout()
plt.show()

# Para evidenciar a diferença, é interessante focar em um detalhe (crop)
# Descomente o código abaixo para ver um zoom de uma área da imagem:
"""
crop_y, crop_x = nova_altura//2, nova_largura//2
tamanho_crop = 200

fig_zoom, (az1, az2) = plt.subplots(1, 2, figsize=(12, 6))
az1.imshow(img_linear[crop_y:crop_y+tamanho_crop, crop_x:crop_x+tamanho_crop])
az1.set_title("Zoom - Linear")
az2.imshow(img_bicubica[crop_y:crop_y+tamanho_crop, crop_x:crop_x+tamanho_crop])
az2.set_title("Zoom - Bicúbica")
plt.show()
"""