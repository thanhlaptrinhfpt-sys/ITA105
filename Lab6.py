import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

image_files = [
    r'C:\Data processing\Labs\Lab 6\anhphongnoithat\i_a.jpg', 
    r'C:\Data processing\Labs\Lab 6\anhphongnoithat\images.jpg', 
    r'C:\Data processing\Labs\Lab 6\anhphongnoithat\tai_xuong.jpg', 
    r'C:\Data processing\Labs\Lab 6\anhphongnoithat\tx_a.jpg', 
    r'C:\Data processing\Labs\Lab 6\anhphongnoithat\tx_b.jpg'
]

def process_lab6(image_path):
   
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không tìm thấy file tại {image_path}")
        return None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    img_resized = cv2.resize(img, (224, 224))
    
    img_norm = img_resized.astype(np.float32) / 255.0
    
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    img_aug = cv2.flip(img_resized, 1)

    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    img_aug = cv2.warpAffine(img_aug, M, (224, 224))
   
    brightness = np.random.uniform(0.8, 1.2)
    img_aug = np.clip(img_aug.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
    return img_resized, img_aug

orig_list = []
aug_list = []

for path in image_files:
    orig, aug = process_lab6(path)
    if orig is not None:
        orig_list.append(orig)
        aug_list.append(aug)

if len(orig_list) == 5:
    plt.figure(figsize=(15, 6))
    for i in range(5):
        # Ảnh gốc
        plt.subplot(2, 5, i + 1)
        plt.imshow(orig_list[i])
        plt.title(f"Gốc {i+1}")
        plt.axis('off')
        
    
        plt.subplot(2, 5, i + 6)
        plt.imshow(aug_list[i])
        plt.title(f"Augmented {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
#===================================================

#2

image_files = [
    r'C:\Data processing\Labs\Lab 6\oto_xemay\x_1.jpg',
    r'C:\Data processing\Labs\Lab 6\oto_xemay\x_2.jpg',
    r'C:\Data processing\Labs\Lab 6\oto_xemay\x_3.jpg',
    r'C:\Data processing\Labs\Lab 6\oto_xemay\x_4.jpg',
    r'C:\Data processing\Labs\Lab 6\oto_xemay\x_5.jpg'
]

def add_gaussian_noise(image):
    """Hàm thêm nhiễu Gaussian vào ảnh"""
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    
    noisy = image.astype(np.float32) + gauss * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)

def process_oto_xemay(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không tìm thấy file tại {image_path}")
        return None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   
    img_resized = cv2.resize(img, (224, 224))
    
  
    img_normalized = img_resized.astype(np.float32) / 255.0

    
    img_aug = add_gaussian_noise(img_resized)
    
   
    brightness = np.random.uniform(0.85, 1.15)
    img_aug = np.clip(img_aug.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
  
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    img_aug = cv2.warpAffine(img_aug, M, (224, 224))

    
    img_aug_gray = cv2.cvtColor(img_aug, cv2.COLOR_RGB2GRAY)
    
    return img_resized, img_aug_gray

orig_list = []
aug_list = []

for path in image_files:
    orig, aug = process_oto_xemay(path)
    if orig is not None:
        orig_list.append(orig)
        aug_list.append(aug)


if len(orig_list) == 5:
    plt.figure(figsize=(15, 7))
    for i in range(5):
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(orig_list[i])
        plt.title(f"Gốc {i+1} (RGB)")
        plt.axis('off')
        
        
        plt.subplot(2, 5, i + 6)
        plt.imshow(aug_list[i], cmap='gray')
        plt.title(f"Aug + Gray {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#=======================================================

#3

image_files = [
    r'C:\Data processing\Labs\Lab 6\traicay\t_1.jpg',
    r'C:\Data processing\Labs\Lab 6\traicay\t_2.jpg',
    r'C:\Data processing\Labs\Lab 6\traicay\t_3.jpg',
    r'C:\Data processing\Labs\Lab 6\traicay\t_4.jpg',
    r'C:\Data processing\Labs\Lab 6\traicay\t_5.jpg'
]

def augment_fruit_image(image):
  
    img = cv2.resize(image, (224, 224))
    
    
    flip_code = random.choice([-1, 0, 1, None])
    if flip_code is not None:
        img = cv2.flip(img, flip_code)
    

    h, w = img.shape[:2]
    crop_size = random.uniform(0.8, 0.95)
    new_h, new_w = int(h * crop_size), int(w * crop_size)
    
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    
    img = img[top:top+new_h, left:left+new_w]
    img = cv2.resize(img, (224, 224))
    
    
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    img = cv2.warpAffine(img, M, (224, 224))
    
 
    img_normalized = img.astype(np.float32) / 255.0
    
    return img_normalized


augmented_pool = []
for path in image_files:
    original = cv2.imread(path)
    if original is not None:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
       
        for _ in range(2): 
            augmented_pool.append(augment_fruit_image(original))


random.shuffle(augmented_pool)
grid_images = augmented_pool[:9]

# Hiển thị grid 3x3
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(grid_images[i])
    plt.title(f"Augmented {i+1}")
    plt.axis('off')

plt.suptitle("Bài tập Lab 6: Fruit Data Augmentation (Grid 3x3)", fontsize=16)
plt.tight_layout()
plt.show()

#====================================================
#4
image_files = [
    r'C:\Data processing\Labs\Lab 6\nha\c_1.jpg',
    r'C:\Data processing\Labs\Lab 6\nha\c_2.jpg',
    r'C:\Data processing\Labs\Lab 6\nha\c_3.jpg',
    r'C:\Data processing\Labs\Lab 6\nha\c_4.jpg',
    r'C:\Data processing\Labs\Lab 6\nha\c_4.jpg' # Lưu ý file c_4 đang bị lặp lại trong danh sách của bạn
]

def augment_image(img_resized):
    
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    img_aug = cv2.warpAffine(img_resized, M, (224, 224))
    
  
    if np.random.rand() > 0.5:
        img_aug = cv2.flip(img_aug, 1)
        

    brightness = np.random.uniform(0.8, 1.2)
    img_aug = np.clip(img_aug.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
   
    img_gray = cv2.cvtColor(img_aug, cv2.COLOR_RGB2GRAY)
    
  
    img_final = img_gray.astype(np.float32) / 255.0
    
    return img_final


for path in image_files:
    img = cv2.imread(path)
    if img is None:
        print(f"Lỗi: Không tìm thấy file {path}")
        continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
   
    aug1 = augment_image(img_resized)
    aug2 = augment_image(img_resized)
    aug3 = augment_image(img_resized)
    
 
    plt.figure(figsize=(12, 3))
    

    plt.subplot(1, 4, 1)
    plt.imshow(img_resized)
    plt.title("Gốc (RGB)")
    plt.axis('off')
    
  
    aug_imgs = [aug1, aug2, aug3]
    for i, aug_img in enumerate(aug_imgs):
        plt.subplot(1, 4, i + 2)
        plt.imshow(aug_img, cmap='gray')
        plt.title(f"Augmented {i+1}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()