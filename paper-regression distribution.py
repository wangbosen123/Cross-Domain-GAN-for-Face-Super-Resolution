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




def PCA_before_x_after_reg():
    global ablation_study_lrec_ladv_enc
    global ablation_study_lrec_ladv_reg
    global ablation_study_lrec_ladv_lreg_enc
    global ablation_study_lrec_ladv_lreg_reg

    ablation_study_lrec_ladv_lreg_enc = encoder()
    ablation_study_lrec_ladv_lreg_reg = regression()
    ablation_study_lrec_ladv_enc = encoder()
    ablation_study_lrec_ladv_reg = regression()

    ablation_study_lrec_ladv_lreg_enc.load_weights('weights/ablation_study_lrec_ladv_lreg_encoder')
    ablation_study_lrec_ladv_lreg_reg.load_weights('weights/ablation_study_lrec_ladv_lreg_reg')

    ablation_study_lrec_ladv_enc.load_weights('weights/ablation_study_lrec_ladv_encoder')
    ablation_study_lrec_ladv_reg.load_weights('weights/ablation_study_lrec_ladv_reg')

    train_path = '/disk2/bosen/Datasets/AR_train/'
    test_path = '/disk2/bosen/Datasets/AR_test/'

    pca_data, pca_1ratio, pca_2ratio, pca_4ratio, pca_8ratio = [], [[] for i in range(3)], [[] for i in range(3)], [[]
                                                                                                                    for
                                                                                                                    i in
                                                                                                                    range(
                                                                                                                        3)], [
        [] for i in range(3)]
    for id in os.listdir(test_path):
        print(int(id[2:]), end=',')
        for file_num, filename in enumerate(os.listdir(test_path + id)):
            if file_num == 30:
                break
            image = cv2.imread(test_path + id + '/' + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

            z, _, _, _, _ = ablation_study_lrec_ladv_enc(image.reshape(1, 64, 64, 1))
            z1, _, _, _, _ = ablation_study_lrec_ladv_enc(low1_image.reshape(1, 64, 64, 1))
            z2, _, _, _, _ = ablation_study_lrec_ladv_enc(low2_image.reshape(1, 64, 64, 1))
            z3, _, _, _, _ = ablation_study_lrec_ladv_enc(low3_image.reshape(1, 64, 64, 1))

            _, _, z = ablation_study_lrec_ladv_reg(z)
            _, _, z1 = ablation_study_lrec_ladv_reg(z1)
            _, _, z2 = ablation_study_lrec_ladv_reg(z2)
            _, _, z3 = ablation_study_lrec_ladv_reg(z3)

            pca_data.append(tf.reshape(z, [200]))
            pca_data.append(tf.reshape(z1, [200]))
            pca_data.append(tf.reshape(z2, [200]))
            pca_data.append(tf.reshape(z3, [200]))
            if (int(id[2:]) == 7):
                plt.imshow(image, cmap='gray')
                plt.show()
                plt.imshow(low1_image, cmap='gray')
                plt.show()
                plt.imshow(low2_image, cmap='gray')
                plt.show()
                plt.imshow(low3_image, cmap='gray')
                plt.show()

            if (int(id[2:]) == 1):
                pca_1ratio[0].append(tf.reshape(z, [200]))
                pca_2ratio[0].append(tf.reshape(z1, [200]))
                pca_4ratio[0].append(tf.reshape(z2, [200]))
                pca_8ratio[0].append(tf.reshape(z3, [200]))
            elif (int(id[2:]) == 2):
                pca_1ratio[1].append(tf.reshape(z, [200]))
                pca_2ratio[1].append(tf.reshape(z1, [200]))
                pca_4ratio[1].append(tf.reshape(z2, [200]))
                pca_8ratio[1].append(tf.reshape(z3, [200]))
            elif (int(id[2:]) == 7):
                pca_1ratio[2].append(tf.reshape(z, [200]))
                pca_2ratio[2].append(tf.reshape(z1, [200]))
                pca_4ratio[2].append(tf.reshape(z2, [200]))
                pca_8ratio[2].append(tf.reshape(z3, [200]))

    # pca_data, pca_1ratio, pca_2ratio, pca_4ratio, pca_8ratio = np.array(pca_data), np.array(pca_1ratio), np.array(pca_2ratio), np.array(pca_4ratio), np.array(pca_8ratio)
    # print(pca_data.shape, pca_1ratio.shape, pca_2ratio.shape, pca_4ratio.shape, pca_8ratio.shape)
    #
    # for id in os.listdir(test_path):
    #     for file_num, filename in enumerate(os.listdir(test_path + id)):
    #         if (int(id[2:])!=1) and (int(id[2:])!=2) and (int(id[2:])!=4):
    #             continue
    #
    #         if file_num == 5:
    #             break
    #         image = cv2.imread(test_path + id + '/' + filename, 0) / 255
    #         image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
    #         blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
    #         low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    #         low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    #         low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    #
    #         z, _, _, _, _ = encoder(image.reshape(1, 64, 64, 1))
    #         z1, _, _, _, _ = encoder(low1_image.reshape(1, 64, 64, 1))
    #         z2, _, _, _, _ = encoder(low2_image.reshape(1, 64, 64, 1))
    #         z3, _, _, _, _ = encoder(low3_image.reshape(1, 64, 64, 1))
    #         # z = AE_enc(image.reshape(1, 64, 64, 1))
    #         # z1 = AE_enc(low1_image.reshape(1, 64, 64, 1))
    #         # z2 = AE_enc(low2_image.reshape(1, 64, 64, 1))
    #         # z3 = AE_enc(low3_image.reshape(1, 64, 64, 1))
    #         _, _, z = reg(z)
    #         _, _, z1 = reg(z1)
    #         _, _, z2 = reg(z2)
    #         _, _, z3 = reg(z3)
    #
    #         if (int(id[2:]) == 1):
    #             pca_1ratio[0].append(tf.reshape(z, [200]))
    #             pca_2ratio[0].append(tf.reshape(z1, [200]))
    #             pca_4ratio[0].append(tf.reshape(z2, [200]))
    #             pca_8ratio[0].append(tf.reshape(z3, [200]))
    #         elif (int(id[2:]) == 2):
    #             pca_1ratio[1].append(tf.reshape(z, [200]))
    #             pca_2ratio[1].append(tf.reshape(z1, [200]))
    #             pca_4ratio[1].append(tf.reshape(z2, [200]))
    #             pca_8ratio[1].append(tf.reshape(z3, [200]))
    #         elif (int(id[2:]) == 4):
    #             pca_1ratio[2].append(tf.reshape(z, [200]))
    #             pca_2ratio[2].append(tf.reshape(z1, [200]))
    #             pca_4ratio[2].append(tf.reshape(z2, [200]))
    #             pca_8ratio[2].append(tf.reshape(z3, [200]))

    pca_data, pca_1ratio, pca_2ratio, pca_4ratio, pca_8ratio = np.array(pca_data), np.array(pca_1ratio), np.array(
        pca_2ratio), np.array(pca_4ratio), np.array(pca_8ratio)
    print(pca_data.shape, pca_1ratio.shape, pca_2ratio.shape, pca_4ratio.shape, pca_8ratio.shape)

    def visualize_pca(data, ratio1, ratio2, ratio4, ratio8):
        # 建立 PCA 模型，設定要保留的主成分數量
        pca = PCA(n_components=2)

        # 訓練 PCA 模型
        pca_result = pca.fit_transform(data)

        # 繪製 PCA 投影
        plt.figure(figsize=(10, 6))

        # 繪製灰色點表示原始資料
        plt.scatter(pca_result[:, 0], pca_result[:, 1], color='grey', alpha=0.5, label='Original Data')

        # 繪製不同顏色的點表示額外的向量投影
        for i, (data1, data2, data3, data4) in enumerate(zip(ratio1, ratio2, ratio4, ratio8)):
            data1_result = pca.transform(data1.reshape(-1, 200))
            data2_result = pca.transform(data2.reshape(-1, 200))
            data3_result = pca.transform(data3.reshape(-1, 200))
            data4_result = pca.transform(data4.reshape(-1, 200))

            if i == 0:
                color = 'blue'
            elif i == 1:
                color = 'red'
            elif i == 2:
                color = 'black'
            plt.scatter(data1_result[:, 0], data1_result[:, 1], label=f'ID {i + 1} 1 Ratio', marker='o', color=color)
            plt.scatter(data2_result[:, 0], data1_result[:, 1], label=f'ID {i + 1} 2 Ratio', marker='x', color=color)
            plt.scatter(data3_result[:, 0], data1_result[:, 1], label=f'ID {i + 1} 4 Ratio', marker='s', color=color)
            plt.scatter(data4_result[:, 0], data1_result[:, 1], label=f'ID {i + 1} 8 Ratio', marker='+', color=color)

        # 添加標題和標籤
        plt.title('PCA Projection of Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # 顯示圖例
        plt.legend()

        # 顯示圖形
        plt.show()

    # 視覺化 PCA 投影
    visualize_pca(pca_data, pca_1ratio, pca_2ratio, pca_4ratio, pca_8ratio)




def motivation_graph():
    path = '/disk2/bosen/Datasets/AR_train/'
    high_images, low_images = [], []
    for id in os.listdir(path):
        for num, filename in enumerate(os.listdir(path + id)):
            if num == 1:
                break
            image = cv2.imread(path + id + '/' + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            high_images.append(image)
            low_images.append(low3_image)
    high_images, low_images = np.array(high_images), np.array(low_images)
    target_image = low_images[12]
    distance = []
    for index, low in enumerate(low_images):
        dis = tf.reduce_mean(tf.square(target_image - low)).numpy()
        distance.append(dis)

    target_high, target_low = [], []
    target_high.append(high_images[12])
    target_low.append(low_images[12])
    distance[distance.index(min(distance))] = max(distance)
    for i in range(5):
        index = distance.index(min(distance))
        distance[distance.index(min(distance))] = max(distance)
        target_high.append(high_images[index])
        target_low.append(low_images[index])
    target_high, target_low = np.array(target_high), np.array(target_low)
    print(target_high.shape, target_low.shape)

    plt.subplots(figsize=(5, 2))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(target_high[i], cmap='gray')
        plt.subplot(2, 5, i + 6)
        plt.axis('off')
        plt.imshow(target_low[i], cmap='gray')
    plt.show()










