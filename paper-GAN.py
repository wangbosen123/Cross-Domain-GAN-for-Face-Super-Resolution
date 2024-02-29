import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from build_model import *
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier




class PatchGAN():
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

        self.reg.load_weights('/disk2/bosen/CDRG-SR/weights/reg_x_cls_REG')
        self.encoder.load_weights('/disk2/bosen/CDRG-SR/weights/encoder')
        self.decoder.load_weights('weights/decoder')
        self.generator.load_weights('weights/unet_generator_20_step4')
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

            z_H, f1_e, f2_e, f3_e, f4_e = self.encoder(high_images)
            syn_image, f1_d, f2_d, f3_d = self.decoder([z_H, f1_e, f2_e, f3_e, f4_e])

            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low_images)
            _, _, zreg = self.reg(z)
            gen_images, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
            fake_score = self.discriminator(gen_images)

            g_loss = tf.reduce_mean(tf.square(fake_score - 1))
            image_loss = 20 * tf.reduce_mean(tf.square(high_images - gen_images))
            style_loss = 5 * self.style_loss(high_images, gen_images)
            dis_loss = tf.reduce_mean(tf.square(f1_d - f1_g)) + tf.reduce_mean(tf.square(f2_d - f2_g)) + tf.reduce_mean(tf.square(f3_d - f3_g))
            total_loss = image_loss + style_loss + g_loss + dis_loss

        if train:
            grads = tape.gradient(total_loss, self.generator.trainable_variables)
            self.g_opti.apply_gradients(zip(grads, self.generator.trainable_variables))
        return image_loss, style_loss, dis_loss, g_loss

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

    def training(self):
        image_loss_epoch = []
        style_loss_epoch = []
        dis_loss_epoch = []
        g_loss_epoch = []
        d_loss_epoch = []

        for epoch in range(1, self.epochs+1):
            start = time.time()
            image_loss_batch = []
            style_loss_batch = []
            dis_loss_batch = []
            g_loss_batch = []
            d_loss_batch = []

            for batch in range(self.batch_num):
                high_images, low_images = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                for i in range(2):
                    d_loss = self.d_train_step(low_images, high_images, train=True)
                d_loss_batch.append(d_loss)

                image_loss, style_loss, dis_loss, g_loss = self.g_train_step(low_images, high_images, train=True)
                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)
                dis_loss_batch.append(dis_loss)
                g_loss_batch.append(g_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            dis_loss_epoch.append(np.mean(dis_loss_batch))
            g_loss_epoch.append(np.mean(g_loss_batch))
            d_loss_epoch.append(np.mean(d_loss_batch))
            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the style_loss is {style_loss_epoch[-1]}')
            print(f'the dis_loss is {dis_loss_epoch[-1]}')
            print(f'the g_loss is {g_loss_epoch[-1]}')
            print(f'the d_loss is {d_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            self.generator.save_weights(f'weights/unet_generator_{epoch}_step{self.step}')

            self.plot_image()
            plt.plot(image_loss_epoch, label='Image Loss')
            plt.title('Image Loss')
            plt.savefig(f'result/generator/image_loss_step{self.step}')
            plt.close()

            plt.plot(style_loss_epoch, label='Style Loss')
            plt.title('Style Loss')
            plt.savefig(f'result/generator/style_loss_step{self.step}')
            plt.close()

            plt.plot(dis_loss_epoch, label='Distillation Loss')
            plt.title('Distillation Loss')
            plt.savefig(f'result/generator/dis_loss_step{self.step}')
            plt.close()

            plt.plot(g_loss_epoch, label='G Loss')
            plt.title('G Loss')
            plt.savefig(f'result/generator/g_loss_step{self.step}')
            plt.close()

            plt.plot(d_loss_epoch, label='D Loss')
            plt.title('D Loss')
            plt.savefig(f'result/generator/d_loss_step{self.step}')
            plt.close()

            plt.plot(g_loss_epoch, label='G Loss')
            plt.plot(d_loss_epoch, label='D Loss')
            plt.legend(['G Loss', 'D Loss'], loc='upper right')
            plt.savefig(f'result/generator/adv_loss_step{self.step}')
            plt.close()

            path = '/disk2/bosen/Datasets/Test/'
            psnr, ssim = [[] for i in range(6)], [[] for i in range(6)]
            pre_psnr, pre_ssim =  [[] for i in range(6)], [[] for i in range(6)]
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
                    syn00, _, _, _ = self.generator([z, f1_e, f2_e, f3_e, f4_e])

                    z, f1_e, f2_e, f3_e, f4_e = self.encoder(low1_image.reshape(1, 64, 64, 1))
                    _, _, zreg = self.reg(z)
                    syn1, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
                    syn11, _, _, _ = self.generator([z, f1_e, f2_e, f3_e, f4_e])

                    z, f1_e, f2_e, f3_e, f4_e = self.encoder(low2_image.reshape(1, 64, 64, 1))
                    _, _, zreg = self.reg(z)
                    syn2, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
                    syn22, _, _, _ = self.generator([z, f1_e, f2_e, f3_e, f4_e])

                    z, f1_e, f2_e, f3_e, f4_e = self.encoder(low3_image.reshape(1, 64, 64, 1))
                    _, _, zreg = self.reg(z)
                    syn3, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
                    syn33, _, _, _ = self.generator([z, f1_e, f2_e, f3_e, f4_e])

                    z, f1_e, f2_e, f3_e, f4_e = self.encoder(low4_image.reshape(1, 64, 64, 1))
                    _, _, zreg = self.reg(z)
                    syn4, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
                    syn44, _, _, _ = self.generator([z, f1_e, f2_e, f3_e, f4_e])

                    z, f1_e, f2_e, f3_e, f4_e = self.encoder(low5_image.reshape(1, 64, 64, 1))
                    _, _, zreg = self.reg(z)
                    syn5, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
                    syn55, _, _, _ = self.generator([z, f1_e, f2_e, f3_e, f4_e])

                    psnr[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
                    psnr[1].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                    psnr[2].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                    psnr[3].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                    psnr[4].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
                    psnr[5].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])
                    pre_psnr[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                 tf.cast(syn00, dtype=tf.float32), max_val=1)[0])
                    pre_psnr[1].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                 tf.cast(syn11, dtype=tf.float32), max_val=1)[0])
                    pre_psnr[2].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                 tf.cast(syn22, dtype=tf.float32), max_val=1)[0])
                    pre_psnr[3].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                 tf.cast(syn33, dtype=tf.float32), max_val=1)[0])
                    pre_psnr[4].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                 tf.cast(syn44, dtype=tf.float32), max_val=1)[0])
                    pre_psnr[5].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                 tf.cast(syn55, dtype=tf.float32), max_val=1)[0])

                    ssim[0].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
                    ssim[1].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                    ssim[2].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                    ssim[3].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                    ssim[4].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
                    ssim[5].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])
                    pre_ssim[0].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                     tf.cast(syn00, dtype=tf.float32), max_val=1)[0])
                    pre_ssim[1].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                     tf.cast(syn11, dtype=tf.float32), max_val=1)[0])
                    pre_ssim[2].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                     tf.cast(syn22, dtype=tf.float32), max_val=1)[0])
                    pre_ssim[3].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                     tf.cast(syn33, dtype=tf.float32), max_val=1)[0])
                    pre_ssim[4].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                     tf.cast(syn44, dtype=tf.float32), max_val=1)[0])
                    pre_ssim[5].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),
                                                     tf.cast(syn55, dtype=tf.float32), max_val=1)[0])

            psnr, ssim = np.array(psnr), np.array(ssim)
            pre_psnr, pre_ssim = np.array(pre_psnr), np.array(pre_ssim)
            psnr, ssim = tf.reduce_mean(psnr, axis=-1), tf.reduce_mean(ssim, axis=-1)
            pre_psnr, pre_ssim = tf.reduce_mean(pre_psnr, axis=-1), tf.reduce_mean(pre_ssim ,axis=-1)
            print(pre_psnr)
            print(pre_ssim)
            print(psnr)
            print(ssim)

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


    gen = PatchGAN(epochs=20, step=4)
    gen.training()



