import numpy as np
seed = 1337  # for reproducibility
# import dlib
import cv2 as cv
import os
import tensorflow as tf
import argparse
# from loss import abs_loss

class ContentNet(object):
    """
    Content net used to generate content image
    Param:
        nb_filter, default 32, filter number used in inception model
        img_size, (width, height)
    Methods:
        build(), build and compile the network
        gen_position_map(img_num), generate the position map of image
        predict(img_path, weight_path), predict the content image given photo path
    """
    def __init__(self, nb_filter=32, img_size=(200, 250)):
        self.nb_filter = nb_filter
        self.width = img_size[0]
        self.height = img_size[1]
        self.build()

    def build(self):
        inception_1 = tf.keras.Sequential()
        inception_2 = tf.keras.Sequential()

        model_1 = tf.keras.Sequential()
        model_3 = tf.keras.Sequential()
        model_5 = tf.keras.Sequential()

        model_1_ = tf.keras.Sequential()
        model_3_ = tf.keras.Sequential()
        model_5_ = tf.keras.Sequential()
        inputs = tf.keras.Input(shape = (self.height, self.width,6))
        inputs1 = tf.keras.Input(shape=(self.height, self.width, 6))
        inputs2 = tf.keras.Input(shape=(self.height, self.width, 6))
        model_1 = tf.keras.layers.Conv2D(self.nb_filter,1,activation='relu',padding='same',input_shape=(self.height, self.width,6))(inputs)
        model_5 = tf.keras.layers.Conv2D(self.nb_filter,5,activation='relu',padding='same',input_shape=(self.height, self.width,6))(inputs1)
        model_3 = tf.keras.layers.Conv2D(self.nb_filter,3,activation='relu',padding='same',input_shape=(self.height, self.width,6))(inputs2)
        # model_1.add(tf.keras.layers.Conv2D(self.nb_filter, 1, activation='relu', padding='same',input_shape=(self.height, self.width, 6)))
        # model_5.add(tf.keras.layers.Conv2D(self.nb_filter, 5, activation='relu', padding='same',input_shape=(self.height, self.width, 6)))
        # model_3.add(tf.keras.layers.Conv2D(self.nb_filter, 3, activation='relu', padding='same',input_shape=(self.height, self.width, 6)))
        concate_1 = tf.keras.layers.concatenate([model_1, model_3, model_5])
        # inception_1 = tf.keras.Model([inputs], concate_1)
        inception_1 = tf.keras.Model([inputs, inputs1, inputs2], concate_1)
        # inception_1 = tf.keras.Model([model_1.input,model_3.input,model_5.input ], concate_1)
        # tf.keras.utils.plot_model(inception_1, "inception_1.png")
        inception_1.summary()
        # inception_2_input_shape = (inception_1.output_shape[1],inception_1.output_shape[2],inception_1.output_shape[3])
        inception_2_input_shape = (inception_1.output_shape[1], inception_1.output_shape[2], inception_1.output_shape[3])
        test1 = tf.keras.layers.Conv2D(self.nb_filter,1,activation='relu',padding='same',input_shape=inception_2_input_shape)(inception_1.output)
        test2 = tf.keras.layers.Conv2D(self.nb_filter,3,activation='relu',padding='same',input_shape=inception_2_input_shape)(inception_1.output)
        test3 = tf.keras.layers.Conv2D(self.nb_filter,5,activation='relu',padding='same',input_shape=inception_2_input_shape)(inception_1.output)
        inception_2 = tf.keras.layers.concatenate([test1, test2, test3])
        batch_norm = tf.keras.layers.BatchNormalization()(inception_2)
        conv1 = tf.keras.layers.Conv2D(128,1,activation='relu',padding='same')(batch_norm)
        conv2 = tf.keras.layers.Conv2D(128,1,activation='relu',padding='same')(conv1)
        conv3 = tf.keras.layers.Conv2D(256,1,activation='relu',padding='same')(conv2)
        batch_norm2 = tf.keras.layers.BatchNormalization()(conv3)
        drop_out = tf.keras.layers.Dropout(0.5)(batch_norm2)
        conv4 = tf.keras.layers.Conv2D(1,3,activation='linear',padding='same')(drop_out)
        conv5 = tf.keras.layers.Conv2D(1,3,activation='linear',padding='same')(conv4)
        out = tf.keras.layers.Reshape((self.height, self.width))(conv5)
        # model = tf.keras.Model([inputs], out)
        model = tf.keras.Model([inputs, inputs1, inputs2], out)
        model.summary()
        self.model = model
        self.result_func = tf.keras.backend.function([self.model.layers[0].input,self.model.layers[1].input,self.model.layers[2].input], self.model.layers[-1].output)

    def gen_position_map(self, img_num=1):
        position_x = range(self.width)
        position_x = np.asarray(position_x)
        position_x = np.reshape(position_x,(1,self.width))
        position_x = np.repeat(position_x,self.height,0)
        position_x = np.reshape(position_x,(1,self.height,self.width))
        position_x = np.repeat(position_x,img_num,0)
        position_x = position_x/ (1. * self.width)

        position_y = range(self.height)
        position_y = np.asarray(position_y)
        position_y = np.reshape(position_y,(self.height,1))
        position_y = np.repeat(position_y,self.width,1)
        position_y = np.reshape(position_y,(1,self.height,self.width))
        position_y = np.repeat(position_y,img_num,0)
        position_y = position_y/ (1. * self.height)

        position_x = np.expand_dims(position_x,-1)
        position_y = np.expand_dims(position_y,-1)
        self.position_x = position_x
        self.position_y = position_y

    def predict(self, img_path, weight_path):
        """
        Predict the content image of given face photo
        Params:
            img_path, path to face photo
            weight_path, path to model weight
        """

        self.gen_position_map()
        img = cv.imread(img_path)
        img = cv.resize(img,(self.width, self.height))
        dog = cal_DOF(img)
        img = img[np.newaxis, ...] / 255.0
        dog = dog[np.newaxis, :, :, np.newaxis]
        self.model.load_weights(weight_path)
        inputs = np.concatenate([img, self.position_x, self.position_y, dog], axis=3)
        results = self.result_func([inputs, inputs, inputs])
        return np.array(results)

def cal_DOF(_img, sigma=2):
    if len(_img.shape) > 2:
        _img = cv.cvtColor(_img, cv.COLOR_RGB2GRAY)
    test = cv.GaussianBlur(_img, (15, 15), sigma)
    DOF = (1.0 * _img - 1.0 * test) / 255.
    return DOF

def abs_loss(y_true, y_pred):
    e_0 = tf.keras.backend.abs(y_pred - y_true)
    return tf.keras.backend.mean(e_0,axis=-1) # + total_variation_loss(y_pred)

def train(data, save_weight_dir, resume, max_epoch=300, img_size=(200, 250), batch_size=8):
    inputs, gros, dogs = data[0], data[1], data[2]
    print("==Start load model==")
    cnet = ContentNet(img_size = img_size)
    print("==Start generate position map==")
    cnet.gen_position_map(img_num = inputs.shape[0])
    inputs = np.concatenate([inputs, cnet.position_x, cnet.position_y, dogs],axis=3)

    if resume:
        cnet.model.load_weights(os.path.join(save_weight_dir, '_inception-snapshot.hdf5'))
    save_best = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_weight_dir, 'inception-best.hdf5'), monitor='val_loss',
                                          verbose=0, save_best_only=True)
    save_snapshot = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_weight_dir, 'inception-snapshot.hdf5'))
    opt = tf.keras.optimizers.Adam(lr=1e-4)
    cnet.model.compile(loss=abs_loss, optimizer=opt)
    cnet.model.fit([inputs,inputs,inputs], gros, batch_size, max_epoch, validation_split=0.1,callbacks=[save_best, save_snapshot],verbose=True)

def generate_train(photo_path, gro_path, size=(200, 250), photo2gray=False):
    inputs = []
    gros = []
    DOFs = []

    for name in sorted(os.listdir(photo_path)):
        if not name.startswith(".") and (name.endswith(".png") or name.endswith(".jpg")) and os.path.isfile(
                os.path.join(photo_path, name)):
            this_img = cv.imread(os.path.join(photo_path, name))
            this_gro = cv.imread(os.path.join(gro_path, name), 0)
            this_gro = cv.resize(this_gro, size)
            this_img = cv.resize(this_img, size)

            gray_img = cv.cvtColor(this_img, cv.COLOR_BGR2GRAY)
            DOF = cal_DOF(gray_img)
            img = gray_img if photo2gray else this_img
            inputs += [img]
            gros += [this_gro]
            DOFs += [DOF]

    inputs = np.array(inputs) / 255.0
    gros = np.array(gros) / 255.0
    dogs = np.array(DOFs)[..., np.newaxis]

    return inputs, gros, dogs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training the content network')
    parser.add_argument('-f', '--facepath', type=str, default='Data/photos', help='Path for training face photos')
    parser.add_argument('-s', '--sketchpath', type=str, default='Data/sketches', help='Path for training sketch images')
    parser.add_argument('--save_weight', type=str, default='Weight/content_weight', help='Path to save content weight')
    parser.add_argument('--resume', type=int, default=0, help='resume the last training')
    parser.add_argument('--minibatch', type=int, default=8)

    arg = parser.parse_args()
    face_path = arg.facepath
    sketch_path = arg.sketchpath
    save_weight_dir = arg.save_weight
    resume = arg.resume
    batch_size = arg.minibatch

    img_size = (200, 250)

    print('===> Generating data to train')
    inputs, gros, dogs = generate_train(face_path, sketch_path, size=img_size)
    print('===> Generated data size [photo, sketch, dog]:', inputs.shape, gros.shape, dogs.shape)
    print('===> Load model and start training')
    train([inputs, gros, dogs], save_weight_dir, resume, batch_size=batch_size)
    # img_size = (200, 250)
    # content_net = ContentNet(img_size=img_size)
    # import tensorflow as tf
    #
    # # Convert the model
    # converter = tf.lite.TFLiteConverter.from_keras_model(content_net.model)
    # tflite_model = converter.convert()
    #
    # # Save the model.
    # with open('model.tflite', 'wb') as f:
    #     f.write(tflite_model)
    # img_dir_path = './test'
    # weight_pathv1 = './Weight/content_weight/inception-snapshot.hdf5'
    # #
    # save_dir = './test'
    # img_path = './Data/photos/00.png'
    # save_pathv1 = './test/c1v1.png'
    # resultv1 = content_net.predict(img_path, weight_pathv1)
    # resultv1 = resultv1.squeeze() * 255
    # cv.imshow('1', resultv1.astype('uint8'))
    # cv.imwrite(save_pathv1, resultv1.astype('uint8'))
    # print( save_pathv1, 'saved')
    # resultv2 = content_net.predict(img_path, weight_pathv1)
    # resultv2 = resultv1.squeeze() * 255
    # cv.imshow('2', resultv2.astype('uint8'))
    # # cv.imwrite(save_pathv2, resultv2.astype('uint8'))
    # # print(save_pathv2, 'saved')
    # resultv3 = content_net.predict(img_path, weight_pathv3)
    # resultv3 = resultv3.squeeze() * 255
    # cv.imshow('3', resultv3.astype('uint8'))
    # # cv.imwrite(save_pathv3, resultv3.astype('uint8'))
    # # print(save_pathv3, 'saved')
    # resultv4 = content_net.predict(img_path, weight_pathv4)
    # resultv4 = resultv4.squeeze() * 255
    # cv.imshow('4', resultv4.astype('uint8'))
    cv.waitKey(0)