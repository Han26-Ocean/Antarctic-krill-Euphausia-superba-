# -*- coding: utf-8 -*-
# Time: 2023/3/7 20:58

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def multi_cnn_model(shape):
    inputs_so = tf.keras.Input(shape)
    inputs_sla = tf.keras.Input(shape)
    inputs_chl = tf.keras.Input(shape)
    inputs_o2 = tf.keras.Input(shape)
    inputs_sst = tf.keras.Input(shape)
    inputs_u_v_value = tf.keras.Input(shape)
    so_1, sla_1, chl_1, o2_1, sst_1, u_v_value_1 = \
        cnn_2d_model(inputs_so), cnn_2d_model(inputs_sla), cnn_2d_model(inputs_chl), \
        cnn_2d_model(inputs_o2), cnn_2d_model(inputs_sst), cnn_2d_model(inputs_u_v_value)

    input_connect = tf.keras.layers.concatenate([so_1, sla_1, chl_1, o2_1, sst_1, u_v_value_1], axis=-1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(input_connect)
    model = tf.keras.Model(
        inputs={
            'so': inputs_so, 'sla': inputs_sla, 'chl': inputs_chl,
            'o2': inputs_o2, 'sst': inputs_sst, 'u_v_value': inputs_u_v_value
        },
        outputs=outputs
    )
    return model

def cnn_2d_model(x):
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='relu')(x)
    return x

def multi_cnn_3d_model(shape_1, shape_2):
    inputs_so = tf.keras.Input(shape_2)
    inputs_sla = tf.keras.Input(shape_2)
    inputs_chl = tf.keras.Input(shape_1)
    inputs_o2 = tf.keras.Input(shape_2)
    inputs_sst = tf.keras.Input(shape_1)
    inputs_u_v_value = tf.keras.Input(shape_1)

    so_1, sla_1, chl_1, o2_1, sst_1, u_v_value_1 = \
        cnn_3d_model(inputs_so), cnn_3d_model(inputs_sla), cnn_3d_model(inputs_chl), \
        cnn_3d_model(inputs_o2), cnn_3d_model(inputs_sst), cnn_3d_model(inputs_u_v_value)

    input_connect = tf.keras.layers.concatenate([so_1, sla_1, chl_1, o2_1, sst_1, u_v_value_1], axis=-1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(input_connect)
    model = tf.keras.Model(
        inputs={
            'sst': inputs_so, 'sstg': inputs_sla, 'chl': inputs_chl,
            'sea_ice': inputs_o2, 'ssh': inputs_sst, 'u_v_value': inputs_u_v_value
        },
        outputs=outputs
    )
    return model

def cnn_3d_model(x):
    time_channel = x.shape[1]
    time_dict = {
        3: [2, 1, 1], 7: [2, 2, 1], 10: [2, 2, 2], 30: [3, 3, 3]
    }
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(time_dict[time_channel][0], 2, 2))(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(time_dict[time_channel][1], 2, 2))(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(time_dict[time_channel][2], 2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='relu')(x)
    return x