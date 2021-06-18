# -*- coding: utf-8 -*-
# import numpy as np
# import tensorflow as tf
# from model import NeuralStyleTransferModel
# import settings
# import utils
# # 创建模型
# model = NeuralStyleTransferModel()
#
# # 加载内容图片
# content_image = utils.load_images(settings.CONTENT_IMAGE_PATH)
# # 风格图片
# style_image = utils.load_images(settings.STYLE_IMAGE_PATH)
#
# # 计算出目标内容图片的内容特征备用
# target_content_features = model([content_image, ])['content']
# # 计算目标风格图片的风格特征
# target_style_features = model([style_image, ])['style']
#
# M = settings.WIDTH * settings.HEIGHT
# N = 3
#
# noise_image = tf.Variable((content_image + np.random.uniform(-0.2, 0.2, (1, settings.HEIGHT, settings.WIDTH, 3))) / 2)
# utils.save_image(noise_image, 'sketch.jpg'.format(settings.OUTPUT_DIR, 20))

from content_model import ContentNet
import cv2 as cv
import tensorflow as tf
img_size = (200, 250)
content_net = ContentNet(img_size=img_size)
#
# # Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(content_net.model)
tflite_model = converter.convert()

# Save the model.
with open('./Weight/content_weight/jsv2/model.tflite', 'wb') as f:
    f.write(tflite_model)

labels = '\n'.join(sorted(content_net.class_indices.keys()))
with open('./Weight/content_weight/jsv2/labels.txt', 'w') as f:
 f.write(labels)


# img_dir_path = './test'
# weight_pathv1 = './Weight/content_weight/inception-best.hdf5'
# #
# save_dir = './test'
#
# img_path = './test/AR_001.png'
# save_pathv1 = './test/c1v1.png'
# resultv1 = content_net.predict(img_path, weight_pathv1)
# resultv1 = resultv1.squeeze() * 255
# cv.imshow('1', resultv1.astype('uint8'))
# cv.waitKey(0)