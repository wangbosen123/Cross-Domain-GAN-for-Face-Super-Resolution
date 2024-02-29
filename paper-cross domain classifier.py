from overall_model import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from collections import Counter

def cls_train(model_type, train):
    global cls
    cls = cls()

    def prepare_data(model):
        global encoder
        global reg
        global gen
        global AE_enc
        global AE_dec
        encoder = encoder()
        reg = regression()
        gen = generator()
        AE_enc = AE_GAN_encoder()
        AE_dec = AE_GAN_decoder()


        if model == 'unet':
            encoder.load_weights('weights/unet_E')
            gen.load_weights('weights/unet_G')
        elif model == 'ae_gan':
            AE_enc.load_weights("/disk2/bosen/CDRG-SR/weights/AE_GAN_E")
            AE_dec.load_weights("/disk2/bosen/CDRG-SR/weights/AE_GAN_G")
        elif model == 'overall_model':
            # encoder.load_weights('/disk2/bosen/CDRG-SR/weights/encoder')
            # reg.load_weights('/disk2/bosen/CDRG-SR/weights/reg_x_cls_REG')
            # gen.load_weights('weights/unet_generator_20_step4')
            encoder.load_weights('weights/ablation_study_lrec_ladv_lreg_encoder')
            reg.load_weights('weights/ablation_study_lrec_ladv_lreg_reg')
            gen.load_weights('weights/ablation_study_lrec_ladv_lreg_generator')


        train_path = '/disk2/bosen/Datasets/AR_train/'
        train_path3 = '/disk2/bosen/Datasets/AR_test/'
        latent, label = [], []

        for id in os.listdir(train_path):
            for file_num, filename in enumerate(os.listdir(train_path + id)):
                if file_num == 10:
                    break
                image = cv2.imread(train_path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

                if model == 'unet':
                    z1, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                    syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                    z2, _, _, _, _ = encoder(syn)
                    latent.append(tf.reshape(z1, [200]))
                    latent.append(tf.reshape(z2, [200]))
                    label.append(tf.one_hot(int(id[2:]) - 1, 111))
                    label.append(tf.one_hot(int(id[2:]) - 1, 111))

                elif model == 'ae_gan':
                    z1 = AE_enc(tf.reshape(image, [1, 64, 64, 1]))
                    latent.append(tf.reshape(z1, [200]))
                    label.append(tf.one_hot(int(id[2:]) - 1, 111))

                elif model == 'overall_model':
                    for num, low in enumerate([image, low1_image, low2_image, low3_image]):
                        z, f1_e, f2_e, f3_e, f4_e = encoder(low.reshape(1, 64, 64, 1))
                        _, _, zreg1 = reg(z)
                        latent.append(tf.reshape(zreg1, [200]))
                        label.append(tf.one_hot(int(id[2:]) - 1, 111))


        for id in os.listdir(train_path3):
            for file_num, filename in enumerate(os.listdir(train_path3 + id)):
                if 20 <= file_num < 40:
                    break
                image = cv2.imread(train_path3 + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

                if model == 'unet':
                    z1, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                    syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                    z2, _, _, _, _ = encoder(syn)
                    latent.append(tf.reshape(z1, [200]))
                    latent.append(tf.reshape(z2, [200]))
                    label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))
                    label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))

                elif model == 'ae_gan':
                    z1 = AE_enc(tf.reshape(image, [1, 64, 64, 1]))
                    latent.append(tf.reshape(z1, [200]))
                    label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))

                elif model == 'overall_model':
                    for num, low in enumerate([image, low1_image, low2_image, low3_image]):
                        z, f1_e, f2_e, f3_e, f4_e = encoder(low.reshape(1, 64, 64, 1))
                        _, _, zreg1 = reg(z)
                        latent.append(tf.reshape(zreg1, [200]))
                        label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))



        return np.array(latent), np.array(label)

    def test(model):
        global encoder
        global reg
        global gen
        global AE_enc
        global AE_dec
        encoder = encoder()
        reg = regression()
        gen = generator()
        AE_enc = AE_GAN_encoder()
        AE_dec = AE_GAN_decoder()

        if model == 'unet':
            encoder.load_weights('weights/unet_E')
            gen.load_weights('weights/unet_G')
            cls.load_weights('weights/cls_unet')
        elif model == 'ae_gan':
            AE_enc.load_weights("/disk2/bosen/CDRG-SR/weights/AE_GAN_E")
            AE_dec.load_weights("/disk2/bosen/CDRG-SR/weights/AE_GAN_G")
            cls.load_weights('weights/cls_ae_gan')
        elif model == 'overall_model':
            encoder.load_weights('weights/ablation_study_lrec_ladv_lreg_encoder')
            reg.load_weights('weights/ablation_study_lrec_ladv_lreg_reg')
            gen.load_weights('weights/ablation_study_lrec_ladv_lreg_generator')
            cls.load_weights('weights/cls_student')


        # path = '/disk2/bosen/Datasets/AR_aligment_other/'
        path = '/disk2/bosen/Datasets/AR_test/'
        pred1_overall, pred2_overall, pred3_overall, pred4_overall, pred5_overall, label = [], [], [], [], [], []
        pred6_overall, pred7_overall, pred8_overall, pred9_overall, pred10_overall, pred11_overall, pred12_overall = [], [], [], [], [], [], []
        for id in os.listdir(path):
            for file_num, filename in enumerate(os.listdir(path + id)):
                    if file_num == 20:
                        break
                    image = cv2.imread(path + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    blur = cv2.GaussianBlur(image, (7, 7), 0)
                    low1_image = cv2.resize(cv2.resize(blur, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(blur, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(blur, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low4_image = cv2.resize(cv2.resize(blur, (11, 11), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low5_image = cv2.resize(cv2.resize(blur, (11, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low6_image = cv2.resize(cv2.resize(blur, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low7_image = cv2.resize(cv2.resize(blur, (10, 9), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low8_image = cv2.resize(cv2.resize(blur, (9, 9), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low9_image = cv2.resize(cv2.resize(blur, (9, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low10_image = cv2.resize(cv2.resize(blur, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low11_image = cv2.resize(cv2.resize(blur, (8, 7), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low12_image = cv2.resize(cv2.resize(blur, (6, 6), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    label.append(int(id[2:]) - 1 + 90)

                    if model == 'unet':
                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low1_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred1 = cls(z1)
                        pred1_overall.append(np.argmax(pred1, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low2_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred2 = cls(z1)
                        pred2_overall.append(np.argmax(pred2, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low3_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred3_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low4_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred4_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low5_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred5_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low6_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred6_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low7_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred7_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low8_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred8_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low9_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred9_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low10_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred10_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low11_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred11_overall.append(np.argmax(pred3, axis=-1)[0])

                        z1, f1_e, f2_e, f3_e, f4_e = encoder(low12_image.reshape(1, 64, 64, 1))
                        # syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                        # z1, _, _, _, _ = encoder(syn)
                        pred3 = cls(z1)
                        pred12_overall.append(np.argmax(pred3, axis=-1)[0])


                    elif model == 'overall_model':
                        z, f1_e, f2_e, f3_e, f4_e = encoder(low1_image.reshape(1, 64, 64, 1))
                        # _, _, z = reg(z)
                        # syn, _, _, _ = gen([z, f1_e, f2_e, f3_e, f4_e])
                        # z, f1_e, f2_e, f3_e, f4_e = encoder(syn)
                        # _, _, z = reg(z)
                        pred1 = cls(z)
                        pred1_overall.append(np.argmax(pred1, axis=-1)[0])

                        z, f1_e, f2_e, f3_e, f4_e = encoder(low2_image.reshape(1, 64, 64, 1))
                        # _, _, z = reg(z)
                        # syn, _, _, _ = gen([z, f1_e, f2_e, f3_e, f4_e])
                        # z, f1_e, f2_e, f3_e, f4_e = encoder(syn)
                        # _, _, z = reg(z)
                        pred2 = cls(z)
                        pred2_overall.append(np.argmax(pred2, axis=-1)[0])

                        z, f1_e, f2_e, f3_e, f4_e = encoder(low3_image.reshape(1, 64, 64, 1))
                        # _, _, z = reg(z)
                        # syn, _, _, _ = gen([z, f1_e, f2_e, f3_e, f4_e])
                        # z, f1_e, f2_e, f3_e, f4_e = encoder(syn)
                        # _, _, z = reg(z)
                        pred3 = cls(z)
                        pred3_overall.append(np.argmax(pred3, axis=-1)[0])

                        z, f1_e, f2_e, f3_e, f4_e = encoder(low4_image.reshape(1, 64, 64, 1))
                        # _, _, z = reg(z)
                        # syn, _, _, _ = gen([z, f1_e, f2_e, f3_e, f4_e])
                        # z, f1_e, f2_e, f3_e, f4_e = encoder(syn)
                        # _, _, z = reg(z)
                        pred4 = cls(z)
                        pred4_overall.append(np.argmax(pred4, axis=-1)[0])

                        z, f1_e, f2_e, f3_e, f4_e = encoder(low5_image.reshape(1, 64, 64, 1))
                        # _, _, z = reg(z)
                        # syn, _, _, _ = gen([z, f1_e, f2_e, f3_e, f4_e])
                        # z, f1_e, f2_e, f3_e, f4_e = encoder(syn)
                        # _, _, z = reg(z)
                        pred5 = cls(z)
                        pred5_overall.append(np.argmax(pred5, axis=-1)[0])

        print(accuracy_score(label, pred1_overall))
        print(accuracy_score(label, pred2_overall))
        print(accuracy_score(label, pred3_overall))
        print(accuracy_score(label, pred4_overall))
        print(accuracy_score(label, pred5_overall))
        print(accuracy_score(label, pred6_overall))
        print(accuracy_score(label, pred7_overall))
        print(accuracy_score(label, pred8_overall))
        print(accuracy_score(label, pred9_overall))
        print(accuracy_score(label, pred10_overall))
        print(accuracy_score(label, pred11_overall))
        print(accuracy_score(label, pred12_overall))



    if train:
        latent, label = prepare_data(model=model_type)
        print(latent.shape, label.shape)
        cls.compile(optimizer='adam',  loss='categorical_crossentropy', metrics=['accuracy'])
        cls.fit(latent, label, epochs=60, batch_size=111)
        cls.save_weights(f'weights/cls_teacher_diversity')
    else:
        test(model=model_type)




def knn(type='overall_model'):
    global encoder
    global reg
    global gen
    global cls
    encoder = encoder()
    reg = regression()
    gen = generator()
    cls = cls()

    if type == 'unet':
        encoder.load_weights('weights/unet_E')
        gen.load_weights('weights/unet_G')
        cls.load_weights('weights/cls_unet')
    else:
        encoder.load_weights('/disk2/bosen/CDRG-SR/weights/encoder')
        reg.load_weights('/disk2/bosen/CDRG-SR/weights/reg_x_cls_REG')
        gen.load_weights('weights/unet_generator_20_step4')
        cls.load_weights('weights/cls_reg')

    def prepare_data(unet=True):
        train_path = '/disk2/bosen/Datasets/AR_train/'
        train_path2 = '/disk2/bosen/Datasets/Train/'
        train_path3 = '/disk2/bosen/Datasets/AR_test/'
        train_path4 = '/disk2/bosen/Datasets/Test/'
        latent, label = [], []

        for id in os.listdir(train_path):
            for file_num, filename in enumerate(os.listdir(train_path + id)):
                if file_num == 10:
                    break
                image = cv2.imread(train_path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                if unet:
                    z1, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                    _, feature1 = cls(z1)
                    syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                    z2, _, _, _, _ = encoder(syn)
                    _, feature2 = cls(z2)
                    latent.append(tf.reshape(feature1, [128]))
                    latent.append(tf.reshape(feature2, [128]))
                    label.append(int(id[2:]))
                    label.append(int(id[2:]))

                else:
                    z, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                    _, _, zreg1 = reg(z)
                    _, feature1 = cls(zreg1)
                    syn, _, _, _ = gen([zreg1, f1_e, f2_e, f3_e, f4_e])
                    z, _, _, _, _ = encoder(syn)
                    _, _, zreg2 = reg(z)
                    _, feature2 = cls(zreg2)
                    latent.append(tf.reshape(feature1, [128]))
                    latent.append(tf.reshape(feature2, [128]))
                    label.append(int(id[2:]))
                    label.append(int(id[2:]))

        for id in os.listdir(train_path2):
            image = cv2.imread(train_path2 + id, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

            if unet:
                z1, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                _, feature1 = cls(z1)
                syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                z2, _, _, _, _ = encoder(syn)
                _, feature2 = cls(z2)
                latent.append(tf.reshape(feature1, [128]))
                latent.append(tf.reshape(feature2, [128]))
                label.append(int(id[0:2]))
                label.append(int(id[0:2]))

            else:
                z, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                _, _, zreg1 = reg(z)
                _, feature1 = cls(zreg1)
                syn, _, _, _ = gen([zreg1, f1_e, f2_e, f3_e, f4_e])
                z, _, _, _, _ = encoder(syn)
                _, _, zreg2 = reg(z)
                _, feature2 = cls(zreg2)
                latent.append(tf.reshape(feature1, [128]))
                latent.append(tf.reshape(feature2, [128]))
                label.append(int(id[0:2]))
                label.append(int(id[0:2]))

        for id in os.listdir(train_path3):
            for file_num, filename in enumerate(os.listdir(train_path3 + id)):
                if file_num == 10:
                    break
                image = cv2.imread(train_path3 + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                if unet:
                    z1, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                    _, feature1 = cls(z1)
                    syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                    z2, _, _, _, _ = encoder(syn)
                    _, feature2 = cls(z2)
                    latent.append(tf.reshape(feature1, [128]))
                    latent.append(tf.reshape(feature2, [128]))
                    label.append(int(id[2:])+90)
                    label.append(int(id[2:])+90)
                else:
                    z, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                    _, _, zreg1 = reg(z)
                    _, feature1 = cls(zreg1)
                    syn, _, _, _ = gen([zreg1, f1_e, f2_e, f3_e, f4_e])
                    z, _, _, _, _ = encoder(syn)
                    _, _, zreg2 = reg(z)
                    _, feature2 = cls(zreg2)
                    latent.append(tf.reshape(feature1, [128]))
                    latent.append(tf.reshape(feature2, [128]))
                    label.append(int(id[2:]) + 90)
                    label.append(int(id[2:]) + 90)

        for id in os.listdir(train_path4):
            image = cv2.imread(train_path4 + id, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

            if unet:
                z1, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                _, feature1 = cls(z1)
                syn, _, _, _ = gen([z1, f1_e, f2_e, f3_e, f4_e])
                z2, _, _, _, _ = encoder(syn)
                _, feature2 = cls(z2)
                latent.append(tf.reshape(feature1, [128]))
                latent.append(tf.reshape(feature2, [128]))
                label.append(int(id[0:2]))
                label.append(int(id[0:2]))

            else:
                z, f1_e, f2_e, f3_e, f4_e = encoder(image.reshape(1, 64, 64, 1))
                _, _, zreg1 = reg(z)
                _, feature1 = cls(zreg1)
                syn, _, _, _ = gen([zreg1, f1_e, f2_e, f3_e, f4_e])
                z, _, _, _, _ = encoder(syn)
                _, _, zreg2 = reg(z)
                _, feature2 = cls(zreg2)
                latent.append(tf.reshape(feature1, [128]))
                latent.append(tf.reshape(feature2, [128]))
                label.append(int(id[0:2]))
                label.append(int(id[0:2]))

        return np.array(latent), np.array(label)


    path = '/disk2/bosen/Datasets/AR_aligment_other/'
    database, data_label = prepare_data(unet=False)
    database, data_label = np.array(database), np.array(data_label)

    acc2ratio, acc4ratio, acc8ratio = 0, 0, 0
    count = 0
    for id in os.listdir(path):
        for num, filename in enumerate(os.listdir(path + id)):
            if '-1-' in filename or '-14-' in filename:
                # if 10 < num <= 30:
                count += 1
                image = cv2.imread( path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)


                # test 2 ratio Accuracy
                z, f1_e, f2_e, f3_e, f4_e = encoder(low1_image.reshape(1, 64, 64, 1))
                _, _, zreg = reg(z)
                syn, _, _, _ = gen([zreg, f1_e, f2_e, f3_e, f4_e])
                z, _, _, _, _ = encoder(syn)
                _, _, zreg = reg(z)
                _, feature = cls(zreg)
                feature = tf.tile(feature, [database.shape[0], 1]).numpy()
                distance = tf.reduce_mean(tf.square(database - feature), axis=-1).numpy()
                distance = list(distance)

                pred_label = []
                for i in range(3):
                    pred_label.append(data_label[distance.index(min(distance))])
                    distance[distance.index(min(distance))] = max(distance)

                counts = Counter(pred_label)
                most_common_pred = max(counts, key=counts.get)
                if int(id[2: ]) == most_common_pred:
                    acc2ratio += 1

                # dot_product = np.sum(database * feature, axis=-1)
                # norm_database = np.linalg.norm(database, axis=-1)
                # norm_feature = np.linalg.norm(feature, axis=-1)
                # cosine_similarity = dot_product / (norm_database * norm_feature)
                # cosine_similarity = list(cosine_similarity)
                # if int(id[2: ]) == data_label[cosine_similarity.index(max(cosine_similarity))]:
                #     acc2ratio += 1


                # test 4 ratio Accuracy
                z, f1_e, f2_e, f3_e, f4_e = encoder(low2_image.reshape(1, 64, 64, 1))
                _, _, zreg = reg(z)
                syn, _, _, _ = gen([zreg, f1_e, f2_e, f3_e, f4_e])
                z, _, _, _, _ = encoder(syn)
                _, _, zreg = reg(z)
                _, feature = cls(zreg)
                feature = tf.tile(feature, [database.shape[0], 1]).numpy()
                distance = tf.reduce_mean(tf.square(database - feature), axis=-1).numpy()
                distance = list(distance)
                pred_label = []
                for i in range(3):
                    pred_label.append(data_label[distance.index(min(distance))])
                    distance[distance.index(min(distance))] = max(distance)

                counts = Counter(pred_label)
                most_common_pred = max(counts, key=counts.get)
                if int(id[2:]) == most_common_pred:
                    acc4ratio += 1
                # feature = tf.tile(feature, [database.shape[0], 1]).numpy()
                # dot_product = np.sum(database * feature, axis=-1)
                # norm_database = np.linalg.norm(database, axis=-1)
                # norm_feature = np.linalg.norm(feature, axis=-1)
                # cosine_similarity = dot_product / (norm_database * norm_feature)
                # cosine_similarity = list(cosine_similarity)
                # if int(id[2:]) == data_label[cosine_similarity.index(max(cosine_similarity))]:
                #     acc4ratio += 1


                # test 8 ratio Accuracy
                z, f1_e, f2_e, f3_e, f4_e = encoder(low3_image.reshape(1, 64, 64, 1))
                _, _, zreg = reg(z)
                syn, _, _, _ = gen([zreg, f1_e, f2_e, f3_e, f4_e])
                z, _, _, _, _ = encoder(syn)
                _, _, zreg = reg(z)
                _, feature = cls(zreg)
                feature = tf.tile(feature, [database.shape[0], 1]).numpy()
                distance = tf.reduce_mean(tf.square(database - feature), axis=-1).numpy()
                distance = list(distance)
                pred_label = []
                for i in range(3):
                    pred_label.append(data_label[distance.index(min(distance))])
                    distance[distance.index(min(distance))] = max(distance)

                counts = Counter(pred_label)
                most_common_pred = max(counts, key=counts.get)
                if int(id[2:]) == most_common_pred:
                    acc8ratio += 1
                # feature = tf.tile(feature, [database.shape[0], 1]).numpy()
                # dot_product = np.sum(database * feature, axis=-1)
                # norm_database = np.linalg.norm(database, axis=-1)
                # norm_feature = np.linalg.norm(feature, axis=-1)
                # cosine_similarity = dot_product / (norm_database * norm_feature)
                # cosine_similarity = list(cosine_similarity)
                # if int(id[2:]) == data_label[cosine_similarity.index(max(cosine_similarity))]:
                #     acc8ratio += 1

    acc2ratio = acc2ratio / count
    acc4ratio = acc4ratio / count
    acc8ratio = acc8ratio / count
    print(count)

    print(acc2ratio, acc4ratio, acc8ratio)




cls_train(model_type='overall_model', train=True)









