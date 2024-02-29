import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from build_model import *
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier




class end_to_end_training():
    def __init__(self, epochs, step):
        #set parameters
        self.epochs = epochs
        self.step = step
        self.g_opti = tf.keras.optimizers.Adam(3e-5)
        self.d_opti = tf.keras.optimizers.Adam(3e-5)

        #set the model
        self.encoder = encoder()
        self.decoder = decoder()
        self.reg = regression()
        self.generator = generator()
        self.discriminator = discriminator()
        self.cls_reg = cls()

        # self.reg.load_weights('/disk2/bosen/CDRG-SR/weights/reg_x_cls_REG')
        # self.encoder.load_weights('/disk2/bosen/CDRG-SR/weights/encoder')
        # self.decoder.load_weights('weights/decoder')
        # self.generator.load_weights('weights/unet_generator_20_step4')
        # self.discriminator.load_weights('/disk2/bosen/CDRG-SR/weights/discriminator2')
        self.reg.load_weights('weights/ablation_study_lrec_ladv_lreg_ldis_reg')
        self.encoder.load_weights('weights/ablation_study_lrec_ladv_lreg_ldis_encoder')
        self.decoder.load_weights('weights/ablation_study_lrec_ladv_lreg_ldis_decoder')
        self.generator.load_weights('weights/ablation_study_lrec_ladv_lreg_ldis_generator')
        self.discriminator.load_weights('weights/ablation_study_lrec_ladv_lreg_ldis_discriminator')
        self.cls_reg.load_weights('weights/cls_reg')

        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        #set the data path
        self.train_path, self.test_path1, self.test_path2, self.test_path3 = self.load_path()
        self.batch_size = 20
        self.batch_num = int(self.train_path.shape[0] / self.batch_size)
        print(self.train_path.shape, self.test_path1.shape, self.test_path2.shape, self.test_path3.shape)

    def load_path(self):
        path_celeba = '/disk2/bosen/Datasets/celeba_train/'
        path_AR_syn_train = '/disk2/bosen/Datasets/AR_train/'
        path_AR_syn_test = '/disk2/bosen/Datasets/AR_test/'
        path_AR_real_train = "/disk2/bosen/Datasets/AR_original_alignment_train90/"
        train_path = []
        test_path1, test_path2, test_path3 = [], [], []
        ID = [f'ID{i}' for i in range(1, 91)]

        for num, filename in enumerate(os.listdir(path_celeba)):
            if num < 2200:
                train_path.append(path_celeba + filename)
            if num < 21:
                test_path3.append(path_celeba + filename)

        for id in ID:
            for num, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if num < 20:
                    train_path.append(path_AR_syn_train + id + '/' + filename)
                if num == 21:
                    test_path2.append(path_AR_syn_train + id + '/' + filename)

        for count, id in enumerate(ID):
            for num, filename in enumerate(os.listdir(path_AR_real_train + id)):
                if '-1-0' in filename or '-1-1' in filename or '-1-2' in filename:
                    train_path.append(path_AR_real_train + id + '/' + filename)

        for ID in os.listdir(path_AR_syn_test):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if num == 1:
                    test_path1.append(path_AR_syn_test + ID + '/' + filename)


        train_path, test_path1, test_path2, test_path3 = np.array(train_path), np.array(test_path1), np.array(test_path2), np.array(test_path3)
        np.random.shuffle(train_path)
        return train_path, test_path1, test_path2, test_path3

    def get_batch_data(self, data, batch_idx, batch_size):
        train_images, ground_truth = [], []
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        for path in train_data:
            image = cv2.imread(path, 0) / 255
            if "AR" in path:
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

            if self.step == 1:
                train_images.append(image)
                ground_truth.append(image)
            if self.step == 2:
                train_images.append(image)
                train_images.append(low1_image)
                ground_truth.append(image)
                ground_truth.append(image)
            if self.step == 3:
                train_images.append(image)
                train_images.append(low1_image)
                train_images.append(low2_image)
                ground_truth.append(image)
                ground_truth.append(image)
                ground_truth.append(image)
            if self.step == 4:
                train_images.append(image)
                train_images.append(low1_image)
                train_images.append(low2_image)
                train_images.append(low3_image)
                ground_truth.append(image)
                ground_truth.append(image)
                ground_truth.append(image)
                ground_truth.append(image)

        ground_truth = np.array(ground_truth).reshape(-1, 64, 64, 1)
        train_images = np.array(train_images).reshape(-1, 64, 64, 1)
        return ground_truth, train_images

    def style_loss(self, real, fake):
        real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
        real = tf.image.grayscale_to_rgb(real)
        fake = tf.image.grayscale_to_rgb(fake)

        real_feature = self.feature_extraction(real)
        fake_feature = self.feature_extraction(fake)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def g_train_step(self, low_images, high_images, train=True):
        with tf.GradientTape() as tape:
            z_H, f1_e_H, f2_e_H, f3_e_H, f4_e_H = self.encoder(high_images)
            _, f1_d_H, f2_d_H, f3_d_H = self.decoder([z_H, f1_e_H, f2_e_H, f3_e_H, f4_e_H])

            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low_images)
            syn_dec, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])
            zreg_1, zreg_2, zreg_3 = self.reg(z)
            syn_gen, f1_g, f2_g, f3_g = self.generator([zreg_3, f1_e, f2_e, f3_e, f4_e])
            fake_score = self.discriminator(syn_gen)

            ladv_g = tf.reduce_mean(tf.square(fake_score - 1))
            lrec_d = 20 * tf.reduce_mean(tf.square(low_images - syn_dec))
            lrec_g = 20 * tf.reduce_mean(tf.square(high_images - syn_gen))
            lstyle_d = 5 * self.style_loss(high_images, syn_dec)
            lstyle_g = 5 * self.style_loss(high_images, syn_gen)
            lrec_d = lrec_d + lstyle_d
            lrec_g = lrec_g + lstyle_g
            ldis = tf.reduce_mean(tf.square(f1_d_H - f1_g)) + tf.reduce_mean(tf.square(f2_d_H - f2_g)) + tf.reduce_mean(tf.square(f3_d_H - f3_g))
            L_reg_stage1 = (tf.reduce_mean(tf.square(z_H - zreg_1)))
            L_reg_stage2 = (tf.reduce_mean(tf.square(z_H - zreg_2)))
            L_reg_stage3 = (tf.reduce_mean(tf.square(z_H - zreg_3)))
            lreg = 0.1 * (L_reg_stage1 + L_reg_stage2 + L_reg_stage3)
            total_loss = lrec_d + lrec_g + ladv_g + lreg

        if train:
            grads = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables + self.reg.trainable_variables + self.generator.trainable_variables)
            self.g_opti.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables + self.reg.trainable_variables + self.generator.trainable_variables))
        return lrec_d, lrec_g, ladv_g, ldis, lreg

    def d_train_step(self, low_images, high_images, train=True):

        with tf.GradientTape() as tape:
            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low_images)
            _, _, zreg = self.reg(z)
            gen_image, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

            real_score = self.discriminator(high_images)
            fake_score = self.discriminator(gen_image)
            d_loss = (tf.reduce_mean(tf.square(real_score - 1)) + tf.reduce_mean(tf.square(fake_score)))*0.5

        if train:
            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opti.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return d_loss

    def training(self, train):
        if train:
            lrec_d_epoch = []
            lrec_g_epoch = []
            ladv_g_epoch = []
            ladv_d_epoch = []
            lreg_epoch = []
            ldis_epoch = []

            for epoch in range(1, self.epochs+1):
                start = time.time()
                lrec_d_batch = []
                lrec_g_batch = []
                ladv_g_batch = []
                ladv_d_batch = []
                lreg_batch = []
                ldis_batch = []

                for batch in range(self.batch_num):
                    high_images, low_images = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                    for i in range(2):
                        d_loss = self.d_train_step(low_images, high_images, train=True)
                    ladv_d_batch.append(d_loss)

                    lrec_d, lrec_g, ladv_g, ldis, lreg = self.g_train_step(low_images, high_images, train=True)
                    lrec_d_batch.append(lrec_d)
                    lrec_g_batch.append(lrec_g)
                    ladv_g_batch.append(ladv_g)
                    lreg_batch.append(lreg)
                    ldis_batch.append(ldis)

                lrec_d_epoch.append(np.mean(lrec_d_batch))
                lrec_g_epoch.append(np.mean(lrec_g_batch))
                ladv_g_epoch.append(np.mean(ladv_g_batch))
                ladv_d_epoch.append(np.mean(ladv_d_batch))
                lreg_epoch.append(np.mean(lreg_batch))
                ldis_epoch.append(np.mean(ldis_batch))


                print(f'the epoch is {epoch}')
                print(f'the Lrec_d is {lrec_d_epoch[-1]}')
                print(f'the Lrec_g is {lrec_g_epoch[-1]}')
                print(f'the Ladv_G is {ladv_g_epoch[-1]}')
                print(f'the Ladv_D is {ladv_d_epoch[-1]}')
                print(f'the Lreg is {lreg_epoch[-1]}')
                print(f'the Ldis is {ldis_epoch[-1]}')
                print(f'the spend time is {time.time() - start} second')
                print('------------------------------------------------')
                self.encoder.save_weights(f'weights/ablation_study_lrec_ladv_lreg_ldis_encoder')
                self.decoder.save_weights(f'weights/ablation_study_lrec_ladv_lreg_ldis_decoder')
                self.reg.save_weights(f'weights/ablation_study_lrec_ladv_lreg_ldis_reg')
                self.generator.save_weights(f'weights/ablation_study_lrec_ladv_lreg_ldis_generator')
                self.discriminator.save_weights(f'weights/ablation_study_lrec_ladv_lreg_ldis_discriminator')

                plt.plot(lrec_d_epoch, label='Lrec_d')
                plt.title('Lrec_d')
                plt.savefig(f'result/ablation_study_ldis/Lrec_d')
                plt.close()

                plt.plot(lrec_g_epoch, label='Lrec_g')
                plt.title('Lrec_g')
                plt.savefig(f'result/ablation_study_ldis/Lrec_g')
                plt.close()

                plt.plot(ladv_g_epoch, label='Ladv_g')
                plt.title('Ladv_g')
                plt.savefig(f'result/ablation_study_ldis/Ladv_g')
                plt.close()

                plt.plot(ladv_d_epoch, label='Ladv_d')
                plt.title('Ladv_d')
                plt.savefig(f'result/ablation_study_ldis/Ladv_d')
                plt.close()

                plt.plot(lreg_epoch, label='Lreg')
                plt.title('Lreg')
                plt.savefig(f'result/ablation_study_ldis/Lreg')
                plt.close()

                plt.plot(ldis_epoch, label='Ldis')
                plt.title('Ldis')
                plt.savefig(f'result/ablation_study_ldis/Ldis')
                plt.close()

        path = '/disk2/bosen/Datasets/AR_test/'
        preds = [[] for i in range(6)]
        labels = []
        psnr, ssim = [[] for i in range(6)], [[] for i in range(6)]
        for id in os.listdir(path):
            for num, filename in enumerate(os.listdir(path + id)):
                if num == 20:
                    break
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low4_image = cv2.resize(cv2.resize(blur_gray, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low5_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                labels.append(int(id[2:])-1+90)

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(image.reshape(1, 64, 64, 1))
                _, _, zreg = self.reg(z)
                pred = self.cls_reg(zreg)
                preds[0].append(np.argmax(pred, axis=-1)[0])
                syn0, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low1_image.reshape(1, 64, 64, 1))
                _, _, zreg = self.reg(z)
                pred = self.cls_reg(zreg)
                preds[1].append(np.argmax(pred, axis=-1)[0])
                syn1, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low2_image.reshape(1, 64, 64, 1))
                _, _, zreg = self.reg(z)
                pred = self.cls_reg(zreg)
                preds[2].append(np.argmax(pred, axis=-1)[0])
                syn2, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low3_image.reshape(1, 64, 64, 1))
                _, _, zreg = self.reg(z)
                pred = self.cls_reg(zreg)
                preds[3].append(np.argmax(pred, axis=-1)[0])
                syn3, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low4_image.reshape(1, 64, 64, 1))
                _, _, zreg = self.reg(z)
                pred = self.cls_reg(zreg)
                preds[4].append(np.argmax(pred, axis=-1)[0])
                syn4, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low5_image.reshape(1, 64, 64, 1))
                _, _, zreg = self.reg(z)
                pred = self.cls_reg(zreg)
                preds[5].append(np.argmax(pred, axis=-1)[0])
                syn5, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

                psnr[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
                psnr[1].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                psnr[2].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                psnr[3].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                psnr[4].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
                psnr[5].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])

                ssim[0].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
                ssim[1].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                ssim[2].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                ssim[3].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                ssim[4].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
                ssim[5].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])
        psnr, ssim = np.array(psnr), np.array(ssim)
        psnr, ssim = tf.reduce_mean(psnr, axis=-1), tf.reduce_mean(ssim, axis=-1)
        print(psnr)
        print(ssim)
        for i in range(6):
            print(accuracy_score(labels, preds[i]), end=' ,')

    def plot_image(self):
        path = '/disk2/bosen/Datasets/Test/'
        for file_num, filename in enumerate(os.listdir(path)):
            image = cv2.imread(path + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low4_image = cv2.resize(cv2.resize(blur_gray, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low5_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

            z, f1_e, f2_e, f3_e, f4_e = self.encoder(image.reshape(1, 64, 64, 1))
            _, _, zreg = self.reg(z)
            syn0, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low1_image.reshape(1, 64, 64, 1))
            _, _, zreg = self.reg(z)
            syn1, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low2_image.reshape(1, 64, 64, 1))
            _, _, zreg = self.reg(z)
            syn2, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low3_image.reshape(1, 64, 64, 1))
            _, _, zreg = self.reg(z)
            syn3, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low4_image.reshape(1, 64, 64, 1))
            _, _, zreg = self.reg(z)
            syn4, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low5_image.reshape(1, 64, 64, 1))
            _, _, zreg = self.reg(z)
            syn5, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])

            plt.subplots(figsize=(6, 2))
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.subplot(2, 6, 1)
            plt.axis('off')
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 6, 2)
            plt.axis('off')
            plt.imshow(low1_image, cmap='gray')
            plt.subplot(2, 6, 3)
            plt.axis('off')
            plt.imshow(low2_image, cmap='gray')
            plt.subplot(2, 6, 4)
            plt.axis('off')
            plt.imshow(low3_image, cmap='gray')
            plt.subplot(2, 6, 5)
            plt.axis('off')
            plt.imshow(low4_image, cmap='gray')
            plt.subplot(2, 6, 6)
            plt.axis('off')
            plt.imshow(low5_image, cmap='gray')

            plt.subplot(2, 6, 7)
            plt.axis('off')
            plt.imshow(tf.reshape(syn0, [64, 64]), cmap='gray')
            plt.subplot(2, 6, 8)
            plt.axis('off')
            plt.imshow(tf.reshape(syn1, [64, 64]), cmap='gray')
            plt.subplot(2, 6, 9)
            plt.axis('off')
            plt.imshow(tf.reshape(syn2, [64, 64]), cmap='gray')
            plt.subplot(2, 6, 10)
            plt.axis('off')
            plt.imshow(tf.reshape(syn3, [64, 64]), cmap='gray')
            plt.subplot(2, 6, 11)
            plt.axis('off')
            plt.imshow(tf.reshape(syn4, [64, 64]), cmap='gray')
            plt.subplot(2, 6, 12)
            plt.axis('off')
            plt.imshow(tf.reshape(syn5, [64, 64]), cmap='gray')
            plt.savefig(f'result/generator/generator_step{self.step}_plot_image{file_num + 1}')
            plt.close()






if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


    ablation_stydy = end_to_end_training(epochs=10, step=4)
    ablation_stydy.training(train=False)

