import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from build_model import *
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


class decoder_step_by_step():
    def __init__(self, epochs, step):
        #set parameters
        self.epochs = epochs
        self.step = step
        self.opti = tf.keras.optimizers.Adam(1e-4)

        #set the model
        self.encoder = encoder()
        self.decoder = decoder()
        self.encoder.load_weights('/disk2/bosen/CDRG-SR/weights/encoder')
        self.decoder.load_weights('weights/decoder')

        #set the data path
        self.train_path, self.test_path1= self.load_path()
        self.batch_size = 40
        self.batch_num = int(self.train_path.shape[0] / self.batch_size)
        print(self.train_path.shape, self.test_path1.shape)

    def load_path(self):
        path_celeba = '/disk2/bosen/Datasets/celeba_train/'
        path_AR_syn_train = '/disk2/bosen/Datasets/AR_train/'

        train_path = []
        test_path1 = []
        ID = [f'ID{i}' for i in range(1, 91)]

        for num, filename in enumerate(os.listdir(path_celeba)):
            if num < 2200:
                train_path.append(path_celeba + filename)

        for id in ID:
            for num, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if num < 20:
                    train_path.append(path_AR_syn_train + id + '/' + filename)


        train_path, test_path1 = np.array(train_path), np.array(test_path1)
        np.random.shuffle(train_path)
        return train_path, test_path1

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
                train_images.append(low3_image)
                ground_truth.append(low3_image)
            if self.step == 2:
                train_images.append(low3_image)
                train_images.append(low2_image)
                ground_truth.append(low3_image)
                ground_truth.append(low2_image)
            if self.step == 3:
                train_images.append(low3_image)
                train_images.append(low2_image)
                train_images.append(low1_image)
                ground_truth.append(low3_image)
                ground_truth.append(low2_image)
                ground_truth.append(low1_image)
            if self.step == 4:
                train_images.append(low3_image)
                train_images.append(low2_image)
                train_images.append(low1_image)
                train_images.append(image)

                ground_truth.append(low3_image)
                ground_truth.append(low2_image)
                ground_truth.append(low1_image)
                ground_truth.append(image)

        ground_truth = np.array(ground_truth).reshape(-1, 64, 64, 1)
        train_images = np.array(train_images).reshape(-1, 64, 64, 1)
        return ground_truth, train_images

    def train_step(self, low_images, high_images, train=True):
        with tf.GradientTape() as tape:
            z, f1_e, f2_e, f3_e, f4_e = self.encoder(low_images)
            syn_image, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])
            image_loss = 10 * tf.reduce_mean(tf.square(high_images - syn_image))
        if train:
            grads = tape.gradient(image_loss, self.decoder.trainable_variables)
            self.opti.apply_gradients(zip(grads,  self.decoder.trainable_variables))
        return image_loss

    def training(self):
        image_loss_epoch = []

        for epoch in range(1, self.epochs+1):
            start = time.time()
            image_loss_batch = []

            for batch in range(self.batch_num):
                high_images, low_images = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                image_loss = self.train_step(low_images, high_images, train=True)
                image_loss_batch.append(image_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))

            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            self.decoder.save_weights('weights/decoder')
            self.test()
            # self.plot_image()

            plt.plot(image_loss_epoch, label='Image Loss')
            plt.title('Image Loss')
            plt.grid(True)
            plt.savefig(f'result/decoder/image_loss_step{self.step}')
            plt.close()
        self.plot_image()

    def test(self):
        path = '/disk2/bosen/Datasets/AR_test/'
        psnr, ssim = [[] for i in range(4)], [[] for i in range(4)]
        for id in os.listdir(path):
            for file_num, filename in enumerate(os.listdir(path + id)):
                if file_num == 1:
                    break
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (11, 11), 0)

                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(image.reshape(1, 64, 64, 1))
                syn1, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low1_image.reshape(1, 64, 64, 1))
                syn2, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low2_image.reshape(1, 64, 64, 1))
                syn3, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low3_image.reshape(1, 64, 64, 1))
                syn4, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])


                psnr[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                psnr[1].append(tf.image.psnr(tf.cast(tf.reshape(low1_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                psnr[2].append(tf.image.psnr(tf.cast(tf.reshape(low2_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                psnr[3].append(tf.image.psnr(tf.cast(tf.reshape(low3_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])

                ssim[0].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                ssim[1].append(tf.image.ssim(tf.cast(tf.reshape(low1_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                ssim[2].append(tf.image.ssim(tf.cast(tf.reshape(low2_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                ssim[3].append(tf.image.ssim(tf.cast(tf.reshape(low3_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])

        psnr, ssim = np.array(psnr), np.array(ssim)
        psnr, ssim = tf.reduce_mean(psnr, axis=-1), tf.reduce_mean(ssim, axis=-1)
        print(psnr)
        print(ssim)

    def plot_image(self):
        path = '/disk2/bosen/Datasets/Test/'
        plt.subplots(figsize=(4, 2))
        plt.subplots_adjust(wspace=0, hspace=0)
        # for id in os.listdir(path):
        for file_num, filename in enumerate(os.listdir(path )):

                image = cv2.imread(path + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (11, 11), 0)

                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(image.reshape(1, 64, 64, 1))
                syn1, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low1_image.reshape(1, 64, 64, 1))
                syn2, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low2_image.reshape(1, 64, 64, 1))
                syn3, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])

                z, f1_e, f2_e, f3_e, f4_e = self.encoder(low3_image.reshape(1, 64, 64, 1))
                syn4, _, _, _ = self.decoder([z, f1_e, f2_e, f3_e, f4_e])

                plt.subplot(2, 4, 1)
                plt.axis('off')
                plt.imshow(image, cmap='gray')
                plt.subplot(2, 4, 2)
                plt.axis('off')
                plt.imshow(low1_image, cmap='gray')
                plt.subplot(2, 4, 3)
                plt.axis('off')
                plt.imshow(low2_image, cmap='gray')
                plt.subplot(2, 4, 4)
                plt.axis('off')
                plt.imshow(low3_image, cmap='gray')

                plt.subplot(2, 4, 5)
                plt.axis('off')
                plt.imshow(tf.reshape(syn1, [64, 64]), cmap='gray')
                plt.subplot(2, 4, 6)
                plt.axis('off')
                plt.imshow(tf.reshape(syn2, [64, 64]), cmap='gray')
                plt.subplot(2, 4, 7)
                plt.axis('off')
                plt.imshow(tf.reshape(syn3, [64, 64]), cmap='gray')
                plt.subplot(2, 4, 8)
                plt.axis('off')
                plt.imshow(tf.reshape(syn4, [64, 64]), cmap='gray')
                plt.savefig(f'result/decoder/decoder_step{self.step}_plot_image{file_num+1}')
                plt.close()







if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    dec = decoder_step_by_step(epochs=20, step=4)
    dec.training()



