import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from build_model import *
import time
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier




def test(kernel=(11, 11)):
    # Overall structure
    global encoder
    global generator
    global reg
    global AE_enc
    global AE_dec
    global unet_enc
    global unet_dec

    unet_enc = encoder()
    encoder = encoder()
    reg = regression()
    unet_dec = generator()
    generator = generator()
    AE_enc = AE_GAN_encoder()
    AE_dec = AE_GAN_decoder()

    encoder.load_weights('weights/ablation_study_lrec_ladv_lreg_encoder')
    reg.load_weights('weights/ablation_study_lrec_ladv_lreg_reg')
    generator.load_weights('weights/ablation_study_lrec_ladv_lreg_generator')
    AE_enc.load_weights("/disk2/bosen/CDRG-SR/weights/AE_GAN_E")
    AE_dec.load_weights("/disk2/bosen/CDRG-SR/weights/AE_GAN_G")
    unet_enc.load_weights('weights/unet_E')
    unet_dec.load_weights('weights/unet_G')

    for id in range(91, 112):
        plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(wspace=0, hspace=0)
        path = f'/disk2/bosen/Datasets/Test/{id}.bmp'
        image = cv2.imread(path, 0) / 255
        image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
        blur_gray = cv2.GaussianBlur(image, kernel, 0)
        low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1,
                                                                                                                     64,
                                                                                                                     64,
                                                                                                                     1)
        low2_image = cv2.resize(cv2.resize(blur_gray, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1,
                                                                                                                     64,
                                                                                                                     64,
                                                                                                                     1)
        low3_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1,
                                                                                                                     64,
                                                                                                                     64,
                                                                                                                     1)
        low4_image = cv2.resize(cv2.resize(blur_gray, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1,
                                                                                                                     64,
                                                                                                                     64,
                                                                                                                     1)
        low5_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1,
                                                                                                                   64,
                                                                                                                   64,
                                                                                                                   1)

        # 1 row
        for num, low in enumerate([image, low1_image, low2_image, low3_image, low4_image, low5_image]):
            plt.subplot(6, 6, num + 1)
            plt.axis('off')
            plt.imshow(tf.reshape(low, [64, 64]), cmap='gray')

        # 2 row
        for num, low in enumerate([image, low1_image, low2_image, low3_image, low4_image, low5_image]):
            z, f1_e, f2_e, f3_e, f4_e = encoder(tf.reshape(low, [1, 64, 64, 1]))
            syn, _, _, _ = generator([z, f1_e, f2_e, f3_e, f4_e])
            plt.subplot(6, 6, num + 19)
            plt.axis('off')
            plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')

        # 3 row
        for num, low in enumerate([image, low1_image, low2_image, low3_image, low4_image, low5_image]):
            z, f1_e, f2_e, f3_e, f4_e = encoder(tf.reshape(low, [1, 64, 64, 1]))
            _, _, zreg = reg(z)
            syn, _, _, _ = generator([zreg, f1_e, f2_e, f3_e, f4_e])
            plt.subplot(6, 6, num + 25)
            plt.axis('off')
            plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')

        # 4 row
        for num, low in enumerate([image, low1_image, low2_image, low3_image, low4_image, low5_image]):
            z, f1_e, f2_e, f3_e, f4_e = encoder(tf.reshape(low, [1, 64, 64, 1]))
            if num == 0:
                zreg = np.load(f'result/reg_test_gaussian_blur1/ratio1_{id}.npy')
            elif num == 1:
                zreg = np.load(f'result/reg_test_gaussian_blur{kernel[0]}/21testu_ratio2_{id}_{id}.bmp_sample.npy')
            elif num == 2:
                zreg = np.load(f'result/reg_test_gaussian_blur{kernel[0]}/21testu_ratio3.2_{id}_{id}.bmp_sample.npy')
            elif num == 3:
                zreg = np.load(f'result/reg_test_gaussian_blur{kernel[0]}/21testu_ratio4_{id}_{id}.bmp_sample.npy')
            elif num == 4:
                zreg = np.load(f'result/reg_test_gaussian_blur{kernel[0]}/21testu_ratio6.4_{id}_{id}.bmp_sample.npy')
            elif num == 5:
                zreg = np.load(f'result/reg_test_gaussian_blur{kernel[0]}/21testu_ratio8_{id}_{id}.bmp_sample.npy')
            syn, _, _, _ = generator([zreg, f1_e, f2_e, f3_e, f4_e])
            plt.subplot(6, 6, num + 31)
            plt.axis('off')
            plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')

        # 5 row
        for num, low in enumerate([image, low1_image, low2_image, low3_image, low4_image, low5_image]):
            z = AE_enc(tf.reshape(low, [1, 64, 64, 1]))
            syn = AE_dec(z)
            plt.subplot(6, 6, num + 7)
            plt.axis('off')
            plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')

        # 6 row
        for num, low in enumerate([image, low1_image, low2_image, low3_image, low4_image, low5_image]):
            z, f1_e, f2_e, f3_e, f4_e = unet_enc(tf.reshape(low, [1, 64, 64, 1]))
            syn, _, _, _ = unet_dec([z, f1_e, f2_e, f3_e, f4_e])
            plt.subplot(6, 6, num + 13)
            plt.axis('off')
            plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')

        plt.savefig(f'result/reg_test_data/compare_result_ID{id}_kernel_{kernel}')
        plt.close()








