import cv2
import matplotlib.pyplot as plt

from overall_model import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# face recognition at function DBt1 accuracy compare.
# gan in version face recognition at method DBt1_inversion_accuracy_compare.
class reg_test_teacher_student():
    def __init__(self, testing):
        self.encoder = encoder()
        self.reg = regression()
        self.cls = cls()
        self.generator = generator()
        self.discriminator = discriminator()

        self.teacher_model = cls()
        self.student_model = cls()

        self.reg.load_weights('weights/ablation_study_lrec_ladv_lreg_reg')
        self.encoder.load_weights('weights/ablation_study_lrec_ladv_lreg_encoder')
        self.generator.load_weights('weights/ablation_study_lrec_ladv_lreg_generator')
        self.discriminator.load_weights('weights/ablation_study_lrec_ladv_lreg_discriminator')
        # self.teacher_model.load_weights('weights/cls_reg_lrec_ladv_lreg')
        self.teacher_model.load_weights('weights/cls_teacher_diversity')
        self.student_model.load_weights('weights/cls_student')
        self.testing = testing
        self.w1 = 50
        self.w2 = 0.0175
        self.w3 = 3
        # (10, 10), (7, 7) w1=50, w2=0.0175, w3=3
        # (8, 7) w1=25, w2=0.01, w3=3 21.43, 0.698
        #default w1 =50, default w2=0.01, default w3=3

    def down_image(self, image, ratio):
        if ratio == 1:
            return tf.cast(image, dtype=tf.float32)
        elif ratio == 2:
            down_syn = tf.image.resize(image, [32, 32], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 3.2:
            down_syn = tf.image.resize(image, [20, 20], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 4:
            down_syn = tf.image.resize(image, [16, 16], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 6.4:
            down_syn = tf.image.resize(image, [8, 7], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 8:
            down_syn = tf.image.resize(image, [8, 8], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn

    def get_test_data(self, ratio, kernel):
        if self.testing == 'testing2': path = '/disk2/bosen/Datasets/AR_test/'
        elif self.testing == 'testing3': path = '/disk2/bosen/Datasets/AR_aligment_other/'

        # path = os.listdir(path)
        # np.random.shuffle(path)
        # path = path[0: 30]
        # overall_file = [[] for i in range(30)]
        # for num, id in enumerate(path):
        #     for filename in os.listdir('/disk2/bosen/Datasets/AR_aligment_other/' + id):
        #         overall_file[num].append('/disk2/bosen/Datasets/AR_aligment_other/' + id + '/' + filename)
        #     np.random.shuffle(overall_file[num])
        #     overall_file[num] = overall_file[num][0: 2]
        # overall_file = np.array(overall_file).reshape(-1)

        # sample_path = 'result/reg_test_gaussian_blur7/'
        # sample_filename = ['ID39/m-059-3-0.bmp', 'ID44/m-069-3-0.bmp', 'ID84/w-050-3-0.bmp']

        test_high_image, test_low_image, test_z, test_id, test_filename = [], [], [], [], []
        for id in os.listdir(path):
            for num, filename in enumerate(os.listdir(path + id)):
                if num == 20:
                    break
        #             for file_num, filename in enumerate(os.listdir(path + id)):
        # for id_num, filename in enumerate(os.listdir(path)):
        #         if id_num == 1:
        #             break
        # for filename in overall_file:
        #         if file_num == 1:
        #             break
        # for filename in sample_filename:
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                # image = cv2.imread(path + filename, 0)/ 255
        # for filename in os.listdir(sample_path):
        #     if 'testu_ratio2' in filename and 'sample' in filename:
        #         id = filename.split('_')[2]
        #         file_name = filename.split('_')[3]
        #         image = cv2.imread(f'/disk2/bosen/Datasets/AR_aligment_other/ID{id}/{file_name}', 0) / 255
        #         image = cv2.imread(filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, kernel, 0)

                if ratio == 1:
                    low_image = image
                elif ratio == 2:
                    low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 3.2:
                    low_image = cv2.resize(cv2.resize(blur_gray, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 4:
                    low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 6.4:
                    low_image = cv2.resize(cv2.resize(blur_gray, (8, 7), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 8:
                    low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                z, _, _, _, _ = self.encoder(low_image.reshape(1, 64, 64, 1))
                test_high_image.append(image)
                test_low_image.append(low_image)
                test_z.append(z)
                # parts = filename.split('/')
                # test_id.append(tf.one_hot(int(parts[5][2: ]) - 1, 111))
                # test_id.append(int(parts[5][2: ]))
                # test_filename.append(parts[-1])
                # test_id.append(int(filename[0: -4]))
                test_id.append(tf.one_hot(int(id[2:]) - 1, 111))
                # test_id.append(tf.one_hot(int(filename[0:-4]) - 1, 111))
                test_filename.append(filename)
                # test_id.append(tf.one_hot(int(id[2:])-1+90, 111))
                # test_filename.append(file_name)
        test_z, test_high_image, test_low_image, test_id, test_filename = np.array(test_z), np.array(test_high_image), np.array(test_low_image), np.array(test_id), np.array(test_filename)
        print(test_z.shape, test_high_image.shape, test_low_image.shape, test_id.shape, test_filename.shape)
        return test_z, test_high_image, test_low_image, test_id, test_filename

    def rec_loss(self, gt, syn):
        return tf.reduce_mean(tf.square(tf.reshape(tf.cast(gt, dtype=tf.float32), [-1, 64, 64, 1]) - tf.reshape(tf.cast(syn, dtype=tf.float32), [-1, 64, 64, 1])))

    def inversion(self, latent, latent_reg, low_image, ratio, grad=True):
        cce = tf.keras.losses.CategoricalCrossentropy()
        # label = tf.reshape(label, [1, 111])
        with tf.GradientTape(persistent=True) as code_tape:
            code_tape.watch(latent_reg)
            pred_student_low = self.student_model(latent)
            pred_teacher_low = self.teacher_model(latent_reg)
            student_low_label = tf.reshape(tf.one_hot(tf.argmax(pred_student_low, axis=1)[0], 111), [1, 111])
            # teacher_low_label = tf.reshape(tf.one_hot(tf.argmax(pred_teacher_low, axis=1)[0]-1, 111), [1, 111])

            _, f1_e, f2_e, f3_e, f4_e = self.encoder(tf.reshape(low_image, [1, 64, 64, 1]))
            syn_reg, _, _, _ = self.generator([latent_reg, f1_e, f2_e, f3_e, f4_e])
            fake_score = self.discriminator(syn_reg)

            # re_z, f1_e, f2_e, f3_e, f4_e = self.encoder(syn_reg)
            # pred_student_syn = self.student_model(re_z)
            # _, _, re_zreg = self.reg(re_z)
            # pred_teacher_syn = self.teacher_model(re_zreg)

            # 50:0.01:3(best)
            rec_loss = self.w1 * self.rec_loss(low_image, self.down_image(syn_reg, ratio))
            # dis_loss = 0.01 * (tf.square(cce(student_low_label, pred_student_low) - cce(student_low_label, pred_student_syn)) + tf.square(cce(teacher_low_label, pred_teacher_low) - cce(teacher_low_label, pred_teacher_syn)))
            dis_loss = self.w2 * cce(student_low_label, pred_teacher_low)
            adv_loss = self.w3 * tf.reduce_mean(tf.square(1 - fake_score))
            # 50, 0.01, 3

            res_total_loss = rec_loss + dis_loss + adv_loss
            # res_total_loss = rec_loss + adv_loss
            # res_total_loss = rec_loss
            # res_total_loss = dis_loss
            # res_total_loss = adv_loss
            if grad:
                gradient_latent = code_tape.gradient(res_total_loss, latent_reg)
                return gradient_latent, rec_loss, dis_loss, adv_loss
            else:
                return rec_loss, dis_loss

    def reg_inversion(self, ratio, kernel, plot=True):
        test_z, test_high_image, test_low_image, test_id, test_filename = self.get_test_data(ratio=ratio, kernel=kernel)

        zreg_opti = []
        rec_loss_record, dis_loss_record, adv_loss_record = [[] for i in range(test_z.shape[0])], [[] for i in range(test_z.shape[0])], [[] for i in range(test_z.shape[0])]
        update_count = [[0 for i in range(10)] for i in range(test_z.shape[0])]
        # plt.subplots(figsize=(3, 7))
        # plt.subplots_adjust(wspace=0, hspace=0)
        print(kernel[0])
        if ratio < 4 and kernel[0] < 8:
            variable = 0.065
        else:
            variable = 0.01
        for num, (latent, high, low, id, filename) in enumerate(zip(test_z, test_high_image, test_low_image, test_id, test_filename)):
            print(num)
            latent_init = latent
            _, _, latent_reg = self.reg(latent)
            total_loss = 1000
            for step in range(1, 11):
                gradient_latent, rec_loss, dis_loss, adv_loss = self.inversion(latent_init, latent_reg, low, ratio)
                for lr in [(i)*(variable) for i in range(0, 3)]:
                    z_search = latent_reg - (lr * gradient_latent)
                    _, rec_loss, dis_loss, adv_loss = self.inversion(latent_init, z_search, low, ratio)
                    res_total_loss = rec_loss + adv_loss + dis_loss
                    if (res_total_loss) < total_loss:
                        total_loss = res_total_loss
                        final_rec_loss = rec_loss
                        final_adv_loss = adv_loss
                        final_dis_loss = dis_loss
                        latent_reg = z_search
                        update_count[num][step-1] += 1
                rec_loss_record[num].append(final_rec_loss)
                dis_loss_record[num].append(final_dis_loss)
                adv_loss_record[num].append(final_adv_loss)
            zreg_opti.append(latent_reg)

            id = np.argmax(id, axis=-1) + 1
            if kernel == (1, 1): np.save(f'result/reg_test_gaussian_blur1/21testu_ratio{ratio}_{id}_{filename}_sample.npy', latent_reg)
            if kernel == (3, 3): np.save(f'result/reg_test_gaussian_blur3/21testu_ratio{ratio}_{id}_{filename}_sample.npy', latent_reg)
            if kernel == (7, 7): np.save(f'result/reg_test_gaussian_blur7/aug_21testu_ratio87_{id}_{filename}_sample.npy', latent_reg)
            if kernel == (11, 11): np.save(f'result/reg_test_gaussian_blur11/21testu_ratio{ratio}_{id}_{filename}_sample.npy', latent_reg)
        #     plt.subplot(7, 3, num + 1)
        #     plt.axis('off')
        #     plt.imshow(high, cmap='gray')
        #     plt.subplot(7, 3, num + 4)
        #     plt.axis('off')
        #     plt.imshow(low, cmap='gray')
        #     z, f1_e, f2_e, f3_e, f4_e = self.encoder(low.reshape(1, 64, 64, 1))
        #     _, _, zreg = self.reg(z)
        #     syn_z, _, _, _ = self.generator([z, f1_e, f2_e, f3_e, f4_e])
        #     syn_reg, _, _, _ = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
        #     syn_opti, _, _, _ = self.generator([latent_reg, f1_e, f2_e, f3_e, f4_e])
        #     plt.subplot(7, 3, num + 7)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn_z, [64, 64]), cmap='gray')
        #     plt.subplot(7, 3, num + 10)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn_reg, [64, 64]), cmap='gray')
        #     plt.subplot(7, 3, num + 13)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn_opti, [64, 64]), cmap='gray')
        #     plt.subplot(7, 3, num + 16)
        #     plt.axis('off')
        #     diff_syn_reg = abs(tf.reshape(high, [64, 64]).numpy() - tf.reshape(syn_reg, [64, 64]).numpy())
        #     diff_syn_reg_binary = (diff_syn_reg > np.mean(diff_syn_reg) + 0.6 * np.std(diff_syn_reg)).astype(int)
        #     plt.imshow(tf.reshape(diff_syn_reg_binary, [64, 64]), cmap='gray')
        #     plt.subplot(7, 3, num + 19)
        #     plt.axis('off')
        #     diff = abs(tf.reshape(high, [64, 64]).numpy() - tf.reshape(syn_opti, [64, 64]).numpy())
        #     diff_binary = (diff > np.mean(diff_syn_reg) + 0.6 * np.std(diff_syn_reg)).astype(int)
        #     plt.imshow(tf.reshape(diff_binary, [64, 64]), cmap='gray')
        # plt.show()
        # plt.close()

        rec_loss_record, dis_loss_record, adv_loss_record, zreg_opti, update_count = np.array(rec_loss_record), np.array(dis_loss_record), np.array(adv_loss_record), np.array(zreg_opti), np.array(update_count)
        print(rec_loss_record.shape, dis_loss_record.shape, zreg_opti.shape, update_count.shape)
        rec_loss_record, dis_loss_record, adv_loss_record, update_count = tf.reduce_mean(rec_loss_record, axis=0), tf.reduce_mean(dis_loss_record, axis=0), tf.reduce_mean(adv_loss_record, axis=0), tf.reduce_mean(update_count, axis=0)
        print(rec_loss_record, dis_loss_record, adv_loss_record, update_count)

        if plot:
            plt.subplots(figsize=(10, 12))
            plt.subplot(2, 2, 1)
            x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            plt.plot(x_coordinates, rec_loss_record, marker='o')

            plt.title('Rec loss')
            plt.xlabel('Iterate Times')
            plt.ylabel('Mean Loss value')
            plt.xticks(x_coordinates)
            plt.legend(['Rec loss'], loc='upper right')
            # if self.testing == 'testing2':
            #     plt.savefig(f'result/reg_test_data/21id_var_{ratio}_ratio_rec_loss')
            #     plt.close()
            # elif self.testing == 'testing3':
            #     plt.savefig(f'result/reg_test_data/90id_var_{ratio}_ratio_rec_loss_teacher_student')
            #     plt.close()

            plt.subplot(2, 2, 2)
            x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            plt.plot(x_coordinates, adv_loss_record, marker='o')

            plt.title('adv loss')
            plt.xlabel('Iterate Times')
            plt.ylabel('Mean Loss value')
            plt.xticks(x_coordinates)
            plt.legend(['Adv loss'], loc='upper right')
            # if self.testing == 'testing2':
            #     plt.savefig(f'result/reg_test_data/21id_var_{ratio}_ratio_adv_loss')
            #     plt.close()
            # elif self.testing == 'testing3':
            #     plt.savefig(f'result/reg_test_data/90id_var_{ratio}_ratio_adv_loss_teacher_student')
            #     plt.close()

            plt.subplot(2, 2, 3)
            x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            plt.plot(x_coordinates, dis_loss_record, marker='o')
            plt.title('Distillation loss')
            plt.xlabel('Iterate Times')
            plt.ylabel('Mean Loss value')
            plt.xticks(x_coordinates)
            plt.legend(['Dis loss'], loc='upper right')
            # if self.testing == 'testing2':
            #     plt.savefig(f'result/reg_test_data/21id_var_{ratio}_ratio_dis_loss')
            #     plt.close()
            # elif self.testing == 'testing3':
            #     plt.savefig(f'result/reg_test_data/90id_var_{ratio}_ratio_dis_loss_teacher_student')
            #     plt.close()

            plt.subplot(2, 2, 4)
            x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            plt.plot(x_coordinates, update_count, marker='o', label='Update Times')

            plt.xticks(x_coordinates)
            plt.title('update_count')
            plt.xlabel('iterate times')
            plt.ylabel('update times')
            plt.legend(['Update count'], loc='upper right')
            if self.testing == 'testing2':
                plt.savefig(f'result/reg_test_data/21id_var_{int(ratio)}_{kernel}')
                plt.close()
            # elif self.testing == 'testing3':
            #     plt.savefig(f'result/reg_test_data/90id_var_{ratio}_update_count_teacher_student')
            #     plt.close()

        z = test_z
        _, _, zreg = self.reg(tf.reshape(z, [-1, 200]))
        zreg_opti = zreg_opti
        return z, zreg, zreg_opti, test_high_image, test_low_image, test_id

    def psnr_ssim(self, ratio, kernel, plot=False):
        z, zreg, zreg_opti, high_image, low_image, _ = self.reg_inversion(ratio, kernel, plot=plot)
        print(z.shape, zreg.shape, zreg_opti.shape, high_image.shape, low_image.shape)
        mPSNR, mSSIM = [[] for i in range(3)], [[] for i in range(3)]

        # plt.subplots(figsize=(21, 7))
        # plt.subplots_adjust(wspace=0, hspace=0)
        # for num, (latent, latent_reg, latent_reg_opti, high, low) in enumerate(zip(z, zreg, zreg_opti, high_image, low_image)):
        #     _, f1_e, f2_e, f3_e, f4_e = self.encoder(low.reshape(1, 64, 64, 1))
        #     syn, _, _, _ = self.generator([latent, f1_e, f2_e, f3_e, f4_e])
        #     syn_reg, _, _, _  = self.generator([tf.reshape(latent_reg, [-1, 200]), f1_e, f2_e, f3_e, f4_e])
        #     syn_reg_opti, _, _, _  = self.generator([tf.reshape(latent_reg_opti, [-1, 200]), f1_e, f2_e, f3_e, f4_e])
        #     plt.subplot(7, 21, num + 1)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(high, [64, 64]), cmap='gray')
        #
        #     plt.subplot(7, 21, num + 22)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(low, [64, 64]), cmap='gray')
        #
        #     plt.subplot(7, 21, num + 43)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')
        #
        #     plt.subplot(7, 21, num + 64)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn_reg, [64, 64]), cmap='gray')
        #
        #     plt.subplot(7, 21, num + 85)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn_reg_opti, [64, 64]), cmap='gray')
        #
        #     plt.subplot(7, 21, num + 106)
        #     plt.axis('off')
        #     diff_syn_reg = abs(tf.reshape(high, [64, 64]).numpy() - tf.reshape(syn_reg, [64, 64]).numpy())
        #     diff_syn_reg_binary = (diff_syn_reg > np.mean(diff_syn_reg) + 0.5*np.std(diff_syn_reg)).astype(int)
        #     plt.imshow(tf.reshape(diff_syn_reg_binary, [64, 64]), cmap='gray')
        #
        #     plt.subplot(7, 21, num + 127)
        #     plt.axis('off')
        #     diff = abs(tf.reshape(high, [64, 64]).numpy() - tf.reshape(syn_reg_opti, [64, 64]).numpy())
        #     diff_binary = (diff > np.mean(diff_syn_reg) + 0.5*np.std(diff_syn_reg)).astype(int)
        #     plt.imshow(tf.reshape(diff_binary, [64, 64]), cmap='gray')
        #
        #     # if plot:
        #     #     if count == 7:
        #     #         count = 8
        #     #         if self.testing == 'testing2':
        #     #             if kernel == (3, 3):
        #     #                 plt.savefig(f'result/reg_test_gaussian_blur3/teastu_21id_var_{str(int(ratio))}_ratio_result')
        #     #                 plt.close()
        #     #             elif kernel == (7, 7):
        #     #                 plt.savefig(f'result/reg_test_gaussian_blur7/teastu_21id_var_{str(int(ratio))}_ratio_result')
        #     #                 plt.close()
        #     #             elif kernel == (11, 11):
        #     #                 plt.savefig(f'result/reg_test_gaussian_blur11/teastu_21id_var_{str(int(ratio))}_ratio_result')
        #     #                 plt.close()
        #     #         elif self.testing == 'testing3':
        #     #             if kernel == (3, 3):
        #     #                 plt.savefig(
        #     #                     f'result/reg_test_gaussian_blur3/teastu_90id_var_{str(int(ratio))}_ratio_result')
        #     #                 plt.close()
        #     #             elif kernel == (7, 7):
        #     #                 plt.savefig(
        #     #                     f'result/reg_test_gaussian_blur7/teastu_90id_var_{str(int(ratio))}_ratio_result')
        #     #                 plt.close()
        #     #             elif kernel == (11, 11):
        #     #                 plt.savefig( f'result/reg_test_gaussian_blur11/teastu_90id_var_{str(int(ratio))}_ratio_result')
        #     #                 plt.close()
        #
        #     mPSNR[0].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
        #     mPSNR[1].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
        #     mPSNR[2].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg_opti, dtype=tf.float32), max_val=1)[0])
        #     mSSIM[0].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
        #     mSSIM[1].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
        #     mSSIM[2].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg_opti, dtype=tf.float32), max_val=1)[0])

        if plot:
            if self.testing == 'testing2':
                if kernel == (3, 3):
                    plt.savefig(f'result/reg_test_gaussian_blur3/teastu_21id_var_{str(int(ratio))}_ratio_result')
                    plt.close()
                elif kernel == (7, 7):
                    plt.savefig(f'result/reg_test_gaussian_blur7/teastu_21id_var_{str(int(ratio))}_ratio_result')
                    plt.close()
                elif kernel == (11, 11):
                    plt.savefig(f'result/reg_test_gaussian_blur11/teastu_21id_var_{str(int(ratio))}_ratio_result')
                    plt.close()
                elif kernel == (1, 1):
                    plt.savefig(f'result/reg_test_gaussian_blur1/teastu_21id_var_{str(int(ratio))}_ratio_result')
            elif self.testing == 'testing3':
                if kernel == (3, 3):
                    plt.savefig(
                        f'result/reg_test_gaussian_blur3/teastu_90id_var_{str(int(ratio))}_ratio_result')
                    plt.close()
                elif kernel == (7, 7):
                    plt.savefig(
                        f'result/reg_test_gaussian_blur7/teastu_90id_var_{str(int(ratio))}_ratio_result')
                    plt.close()
                elif kernel == (11, 11):
                    plt.savefig(f'result/reg_test_gaussian_blur11/teastu_90id_var_{str(int(ratio))}_ratio_result')
                    plt.close()

        print(mPSNR[0])
        print(mPSNR[1])
        print(mPSNR[2])
        mPSNR = tf.reduce_mean(mPSNR, axis=-1)
        mSSIM = tf.reduce_mean(mSSIM, axis=-1)
        print(mPSNR, mSSIM)

    def psnr_ssim_compare(self, visual=False):
        # path = 'result/reg_test_gaussian_blur7/'
        # psnr_before, psnr_after = [], []
        # ssim_before, ssim_after = [], []
        #
        # for filename in os.listdir(path):
        #     # if 'testu_ratio8' in filename:
        #     if 'ratio4' in filename and 'bmp' in filename and 'testu' not in filename:
        #         print(filename)
        #         latent = np.load(path + filename)
        #         id = filename.split('_')[1]
        #         file_name = filename.split('_')[2][0:-4]
        #         image = cv2.imread(f'/disk2/bosen/Datasets/AR_aligment_other/ID{id}/{file_name}', 0) / 255
        #         image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
        #         blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
        #         low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        #
        #         z, f1_e, f2_e, f3_e, f4_e = self.encoder(tf.reshape(low_image, [1, 64, 64, 1]))
        #         _, _, zreg = self.reg(z)
        #         syn_reg, _, _, _ = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
        #         syn_opti, _, _, _ = self.generator([latent, f1_e, f2_e, f3_e, f4_e])
        #
        #         psnr_after.append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),tf.cast(syn_opti, dtype=tf.float32), max_val=1)[0])
        #         ssim_after.append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32),tf.cast(syn_opti, dtype=tf.float32), max_val=1)[0])
        #         psnr_before.append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
        #         ssim_before.append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
        # print(tf.reduce_mean(psnr_before), tf.reduce_mean(ssim_before))
        # print(tf.reduce_mean(psnr_after), tf.reduce_mean(ssim_after))


        sample_id = ['39', '44', '84']
        sample_filename = ['m-059-3-0', 'm-069-3-0', 'w-050-3-0']
        sample_path = 'result/reg_test_gaussian_blur7/'
        plt.subplots(figsize=(9, 5))
        plt.subplots_adjust(hspace=0, wspace=0)
        for num, (id, filename) in enumerate(zip(sample_id, sample_filename)):
            beforeratio2_latent = np.load(sample_path + 'ratio2_' + id + '_' + filename + '.bmp.npy')
            beforeratio4_latent = np.load(sample_path + 'ratio4_' + id + '_' + filename + '.bmp.npy')
            beforeratio8_latent = np.load(sample_path + 'ratio8_' + id + '_' + filename + '.bmp.npy')
            afterratio2_latent = np.load(sample_path + 'testu_ratio2_' + id + '_' + filename + '.bmp.npy')
            afterratio4_latent = np.load(sample_path + 'testu_ratio4_' + id + '_' + filename + '.bmp.npy')
            afterratio8_latent = np.load(sample_path + 'testu_ratio8_' + id + '_' + filename + '.bmp.npy')
            image = cv2.imread(f'/disk2/bosen/Datasets/AR_aligment_other/ID{id}/{filename}.bmp', 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

            z1, f11_e, f21_e, f31_e, f41_e = self.encoder(tf.reshape(low1_image, [1, 64, 64, 1]))
            z2, f12_e, f22_e, f32_e, f42_e = self.encoder(tf.reshape(low2_image, [1, 64, 64, 1]))
            z3, f13_e, f23_e, f33_e, f43_e = self.encoder(tf.reshape(low3_image, [1, 64, 64, 1]))

            _, _, zreg1 = self.reg(z1)
            _, _, zreg2 = self.reg(z2)
            _, _, zreg3 = self.reg(z3)

            syn_init1, _, _, _ = self.generator([zreg1, f11_e, f21_e, f31_e, f41_e])
            syn_init2, _, _, _ = self.generator([zreg2, f12_e, f22_e, f32_e, f42_e])
            syn_init3, _, _, _ = self.generator([zreg3, f13_e, f23_e, f33_e, f43_e])

            syn_before1, _, _, _ = self.generator([beforeratio2_latent, f11_e, f21_e, f31_e, f41_e])
            syn_before2, _, _, _ = self.generator([beforeratio4_latent, f12_e, f22_e, f32_e, f42_e])
            syn_before3, _, _, _ = self.generator([beforeratio8_latent, f13_e, f23_e, f33_e, f43_e])

            syn_after1, _, _, _ = self.generator([afterratio2_latent, f11_e, f21_e, f31_e, f41_e])
            syn_after2, _, _, _ = self.generator([afterratio4_latent, f12_e, f22_e, f32_e, f42_e])
            syn_after3, _, _, _ = self.generator([afterratio8_latent, f13_e, f23_e, f33_e, f43_e])

            for i in range(1, 4):
                print((num * 3) + i)
                plt.subplot(5, 9, (num * 3) + i)
                plt.axis('off')
                plt.imshow(image, cmap='gray')

            plt.subplot(5, 9, (num * 3) + 10)
            plt.axis('off')
            plt.imshow(low1_image, cmap='gray')
            plt.subplot(5, 9, (num * 3) + 11)
            plt.axis('off')
            plt.imshow(low2_image, cmap='gray')
            plt.subplot(5, 9, (num * 3) + 12)
            plt.axis('off')
            plt.imshow(low3_image, cmap='gray')

            plt.subplot(5, 9, (num * 3) + 19)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_init1, [64, 64]), cmap='gray')
            plt.subplot(5, 9, (num * 3) + 20)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_init2, [64, 64]), cmap='gray')
            plt.subplot(5, 9, (num * 3) + 21)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_init3, [64, 64]), cmap='gray')

            plt.subplot(5, 9, (num * 3) + 28)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_before1, [64, 64]), cmap='gray')
            plt.subplot(5, 9, (num * 3) + 29)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_before2, [64, 64]), cmap='gray')
            plt.subplot(5, 9, (num * 3) + 30)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_before3, [64, 64]), cmap='gray')

            plt.subplot(5, 9, (num * 3) + 37)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_after1, [64, 64]), cmap='gray')
            plt.subplot(5, 9, (num * 3) + 38)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_after2, [64, 64]), cmap='gray')
            plt.subplot(5, 9, (num * 3) + 39)
            plt.axis('off')
            plt.imshow(tf.reshape(syn_after3, [64, 64]), cmap='gray')
        plt.show()

    # def accuracy_compare(self, visual=False):
    #     path = 'result/reg_test_gaussian_blur7/'
    #
    #     student_pred, teacher_pred, opti_pred = [], [], []
    #     true_label = []
    #     for filename in os.listdir(path):
    #         # if 'testu_ratio2' in filename:
    #         if 'ratio2' in filename and 'bmp' in filename and 'testu' not in filename:
    #             print(filename)
    #             latent = np.load(path + filename)
    #             id = filename.split('_')[1]
    #             file_name = filename.split('_')[2][0:-4]
    #             image = cv2.imread(f'/disk2/bosen/Datasets/AR_aligment_other/ID{id}/{file_name}', 0) / 255
    #             image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
    #             blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
    #             low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    #
    #             z, f1_e, f2_e, f3_e, f4_e = self.encoder(tf.reshape(low_image, [1, 64, 64, 1]))
    #             _, _, zreg = self.reg(z)
    #             pred_student, _, _, _ = self.student_model(z)
    #             pred_teacher, _, _, _ = self.teacher_model(zreg)
    #             pred_teacher_opti, _, _, _ = self.teacher_model(latent)
    #             true_label.append(int(id)-1)
    #             student_pred.append(np.argmax(pred_student, axis=-1)[0])
    #             teacher_pred.append(np.argmax(pred_teacher, axis=-1)[0])
    #             opti_pred.append(np.argmax(pred_teacher_opti, axis=-1)[0])
    #     print(accuracy_score(true_label, student_pred))
    #     print(accuracy_score(true_label, teacher_pred))
    #     print(accuracy_score(true_label, opti_pred))


    def DBt1_accuracy_compare(self):
        path = '/disk2/bosen/Datasets/AR_test/'

        high_preds, student_preds, sum_preds, true_label = [[] for i in range(12)], [[] for i in range(12)], [[] for i in range(12)], []

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
                low4_image = cv2.resize(cv2.resize(blur_gray, (11, 11), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low5_image = cv2.resize(cv2.resize(blur_gray, (11, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low6_image = cv2.resize(cv2.resize(blur_gray, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low7_image = cv2.resize(cv2.resize(blur_gray, (10, 9), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low8_image = cv2.resize(cv2.resize(blur_gray, (9, 9), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low9_image = cv2.resize(cv2.resize(blur_gray, (9, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low10_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low11_image = cv2.resize(cv2.resize(blur_gray, (8, 7), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low12_image = cv2.resize(cv2.resize(blur_gray, (7, 7), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)


                true_label.append(int(id[2:]) - 1 + 90)

                for num, low in enumerate([low1_image, low2_image, low3_image, low4_image, low5_image, low6_image, low7_image, low8_image, low9_image, low10_image, low11_image, low12_image]):
                    z_init, f1_e, f2_e, f3_e, f4_e = self.encoder(tf.reshape(low, [1, 64, 64, 1]))
                    _, _, zreg = self.reg(z_init)
                    # syn_gen, f1_g, f2_g, f3_g = self.generator([zreg, f1_e, f2_e, f3_e, f4_e])
                    # z, f1_e, f2_e, f3_e, f4_e = self.encoder(tf.reshape(syn_gen, [1, 64, 64, 1]))
                    # _, _, zreg = self.reg(z)


                    pred_student = self.student_model(z_init)
                    pred = self.teacher_model(zreg)
                    sum_pred = pred_student + pred
                    high_preds[num].append(np.argmax(pred, axis=-1)[0])
                    student_preds[num].append(np.argmax(pred_student, axis=-1)[0])
                    sum_preds[num].append(np.argmax(sum_pred, axis=-1)[0])

        print('student prediction')
        print(f'student prediction (32, 32) size is {accuracy_score(true_label, student_preds[0])}')
        print(f'student prediction (20, 20) size is {accuracy_score(true_label, student_preds[1])}')
        print(f'student prediction (16, 16) size is {accuracy_score(true_label, student_preds[2])}')
        print(f'student prediction (11, 11) size is {accuracy_score(true_label, student_preds[3])}')
        print(f'student prediction (11, 10) size is {accuracy_score(true_label, student_preds[4])}')
        print(f'student prediction (10, 10) size is {accuracy_score(true_label, student_preds[5])}')
        print(f'student prediction (10, 9) size is {accuracy_score(true_label, student_preds[6])}')
        print(f'student prediction (9, 9) size is {accuracy_score(true_label, student_preds[7])}')
        print(f'student prediction (9, 8) size is {accuracy_score(true_label, student_preds[8])}')
        print(f'student prediction (8, 8) size is {accuracy_score(true_label, student_preds[9])}')
        print(f'student prediction (8, 7) size is {accuracy_score(true_label, student_preds[10])}')
        print(f'student prediction (7, 7) size is {accuracy_score(true_label, student_preds[11])}')


        print('------------')
        print('reg prediction')
        print(f'reg prediction (32, 32) size is {accuracy_score(true_label, high_preds[0])}')
        print(f'reg prediction (20, 20) size is {accuracy_score(true_label, high_preds[1])}')
        print(f'reg prediction (16, 16) size is {accuracy_score(true_label, high_preds[2])}')
        print(f'reg prediction (11, 11) size is {accuracy_score(true_label, high_preds[3])}')
        print(f'reg prediction (11, 10) size is {accuracy_score(true_label, high_preds[4])}')
        print(f'reg prediction (10, 10) size is {accuracy_score(true_label, high_preds[5])}')
        print(f'reg prediction (10, 9) size is {accuracy_score(true_label, high_preds[6])}')
        print(f'reg prediction (9, 9) size is {accuracy_score(true_label, high_preds[7])}')
        print(f'reg prediction (9, 8) size is {accuracy_score(true_label, high_preds[8])}')
        print(f'reg prediction (8, 8) size is {accuracy_score(true_label, high_preds[9])}')
        print(f'reg prediction (8, 7) size is {accuracy_score(true_label, high_preds[10])}')
        print(f'reg prediction (7, 7) size is {accuracy_score(true_label, high_preds[11])}')

        print('-------------')
        print('sum of student and reg prediction')
        print(f'sum of student and reg prediction (32, 32) size is {accuracy_score(true_label, sum_preds[0])}')
        print(f'sum of student and reg prediction (20, 20) size is {accuracy_score(true_label, sum_preds[1])}')
        print(f'sum of student and reg prediction (16, 16) size is {accuracy_score(true_label, sum_preds[2])}')
        print(f'sum of student and reg prediction (11, 11) size is {accuracy_score(true_label, sum_preds[3])}')
        print(f'sum of student and reg prediction (11, 10) size is {accuracy_score(true_label, sum_preds[4])}')
        print(f'sum of student and reg prediction (10, 10) size is {accuracy_score(true_label, sum_preds[5])}')
        print(f'sum of student and reg prediction (10, 9) size is {accuracy_score(true_label, sum_preds[6])}')
        print(f'sum of student and reg prediction (9, 9) size is {accuracy_score(true_label, sum_preds[7])}')
        print(f'sum of student and reg prediction (9, 8) size is {accuracy_score(true_label, sum_preds[8])}')
        print(f'sum of student and reg prediction (8, 8) size is {accuracy_score(true_label, sum_preds[9])}')
        print(f'sum of student and reg prediction (8, 7) size is {accuracy_score(true_label, sum_preds[10])}')
        print(f'sum of student and reg prediction (7, 7) size is {accuracy_score(true_label, sum_preds[11])}')


    def DBt1_inversion_accuracy_compare(self):
        high_preds, student_preds, sum_preds, true_label = [[] for i in range(1)], [[] for i in range(1)], [[] for i in range(1)], []
        path = 'result/reg_test_gaussian_blur7/'
        keyword = 'aug_21testu_ratio87'

        count = 0
        PSNR, SSIM = [], []
        for filename in os.listdir(path):
            # if 'aug_21testu_ratio914' in filename:
            #     continue
            if keyword in filename and 'sample' in filename:
                count += 1
                latent = np.load(path + filename)
                id = filename.split('_')[3]
                file_name1 = filename.split('_')[4]
                file_name2 = filename.split('_')[5]
                true_label.append(int(id) - 1 + 90)

                if int(id) < 10:
                    image = cv2.imread(f'/disk2/bosen/Datasets/AR_test/ID0{int(id)}/{file_name1}_{file_name2}', 0) / 255
                else:
                    image = cv2.imread(f'/disk2/bosen/Datasets/AR_test/ID{int(id)}/{file_name1}_{file_name2}',0) / 255

                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low_image = cv2.resize(cv2.resize(blur_gray, (8, 7), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                for num, low in enumerate([low_image]):
                    z_init, f1_e, f2_e, f3_e, f4_e = self.encoder(tf.reshape(low, [1, 64, 64, 1]))
                    _, _, zreg_init = self.reg(z_init)

                    syn_gen, f1_g, f2_g, f3_g = self.generator([latent, f1_e, f2_e, f3_e, f4_e])
                    z, f1_e, f2_e, f3_e, f4_e = self.encoder(tf.reshape(syn_gen, [1, 64, 64, 1]))
                    _, _, zreg = self.reg(z)

                    pred_student = self.student_model(z_init)
                    pred = self.teacher_model(zreg_init)
                    sum_pred = pred_student + pred
                    high_preds[num].append(np.argmax(pred, axis=-1)[0])
                    student_preds[num].append(np.argmax(pred_student, axis=-1)[0])
                    sum_preds[num].append(np.argmax(sum_pred, axis=-1)[0])
                    PSNR.append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_gen, dtype=tf.float32), max_val=1)[0])
                    SSIM.append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_gen, dtype=tf.float32), max_val=1)[0])
#811, 847, 854
        print(len(true_label))
        print('student prediction')
        print(f'student predictions {accuracy_score(true_label, student_preds[0])}')
        print('------------')
        print('reg prediction')
        print(f'reg prediction (32, 32) size is {accuracy_score(true_label, high_preds[0])}')
        print('-------------')
        print('sum of student and reg prediction')
        print(f'sum of student and reg prediction (32, 32) size is {accuracy_score(true_label, sum_preds[0])}')
        print(tf.reduce_mean(PSNR))
        print(tf.reduce_mean(SSIM))







if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)



    reg_test = reg_test_teacher_student(testing='testing2')
    # reg_test.DBt1_accuracy_compare()
    # # reg_test.get_test_data(ratio=2, kernel=(7, 7))
    #
    # reg_test.psnr_ssim(1, kernel=(1, 1))
    # reg_test.psnr_ssim(2, kernel=(3, 3))
    # reg_test.psnr_ssim(2, kernel=(7, 7))
    # reg_test.psnr_ssim(2, kernel=(11, 11))
    #
    # reg_test.psnr_ssim(3.2, kernel=(3, 3))
    # reg_test.psnr_ssim(3.2, kernel=(7, 7))
    # reg_test.psnr_ssim(3.2, kernel=(11, 11))
    #
    # reg_test.psnr_ssim(4, kernel=(3, 3))
    # reg_test.psnr_ssim(4, kernel=(7, 7))
    # reg_test.psnr_ssim(4, kernel=(11, 11))
    #
    # reg_test.psnr_ssim(6.4, kernel=(3, 3))
    # reg_test.psnr_ssim(6.4, kernel=(7, 7))
    reg_test.DBt1_inversion_accuracy_compare()
    # reg_test.DBt1_accuracy_compare()
    # reg_test.psnr_ssim(6.4, kernel=(11, 11))
    #
    # reg_test.psnr_ssim(8, kernel=(3, 3))
    # reg_test.psnr_ssim(8, kernel=(7, 7))
    # reg_test.psnr_ssim(8, kernel=(11, 11))
































