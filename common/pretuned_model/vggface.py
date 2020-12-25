import numpy as np
from keras_applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf

__TF_VERSION__ = int(tf.__version__.split(".")[0])

if __TF_VERSION__ < 2:
    from tensorflow.python.keras import backend as K
else:
    from tensorflow.keras import backend as K


VGG16_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
VGG16_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5'
VGG16_WEIGHTS_PATH_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
VGGFACE_DIR = 'models/vggface'


def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def VGGFace(include_top=True,
            input_tensor=None, 
            input_shape=None,
            pooling=None,
            classes=2622):
            
    input_tensor *= 255
    shape = input_tensor.shape.as_list()
    if shape[1:] != [224, 224, 3]:
        assert shape[-1] == 3
        input_tensor = tf.image.resize(input_tensor, [224, 224])
    input_tensor = tf.numpy_function(preprocess_input, [input_tensor], tf.float32)
    input_tensor.set_shape(shape)
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    if include_top:
        # Classification block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, name='fc6')(x)
        x = tf.keras.layers.Activation('relu', name='fc6/relu')(x)
        x = tf.keras.layers.Dense(4096, name='fc7')(x)
        x = tf.keras.layers.Activation('relu', name='fc7/relu')(x)
        x = tf.keras.layers.Dense(classes, name='fc8')(x)
        x = tf.keras.layers.Activation('softmax', name='fc8/softmax')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.MaxPooling2D()(x)

    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    model = tf.keras.Model(inputs, x, name='vggface_vgg16')
    if not include_top:
        weights_path = tf.keras.utils.get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                                VGG16_WEIGHTS_PATH_NO_TOP,
                                cache_subdir=VGGFACE_DIR)
    else:
        weights_path = tf.keras.utils.get_file('rcmalli_vggface_tf_top_vgg16.h5',
                                VGG16_WEIGHTS_PATH_TOP,
                                cache_subdir=VGGFACE_DIR)   
    model.load_weights(weights_path, by_name=True)
    return model


def VGGFace_block(input_tensor=None, block="conv1", classes=2622, input_shape=None):
    if block == "conv1":
        assert input_tensor is not None
        input_tensor *= 255
        shape = input_tensor.shape.as_list()
        if shape[1:] != [224, 224, 3]:
            assert shape[-1] == 3
            input_tensor = tf.image.resize(input_tensor, [224, 224])
        input_tensor = tf.numpy_function(preprocess_input, [input_tensor], tf.float32)
        input_tensor.set_shape(shape)
        input_shape = _obtain_input_shape(input_shape,
                                        default_size=224,
                                        min_size=48,
                                        data_format=K.image_data_format(),
                                        require_flatten=False)

        if input_tensor is None:
            input_ = tf.keras.layers.Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                input_ = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                input_ = input_tensor

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(input_)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    else:
        if input_tensor is None:
            input_ = tf.keras.layers.Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                input_ = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                input_ = input_tensor

        if block == "conv2":
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(input_)
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        elif block == "conv3":
            x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(input_)
            x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
            x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)        

        elif block == "conv4":
            x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(input_)
            x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
            x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

        elif block == "conv5":
            x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(input_)
            x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
            x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

        elif block == "top":
            # Classification block
            x = tf.keras.layers.Flatten(name='flatten')(input_)
            x = tf.keras.layers.Dense(4096, name='fc6')(x)
            x = tf.keras.layers.Activation('relu', name='fc6/relu')(x)
            x = tf.keras.layers.Dense(4096, name='fc7')(x)
            x = tf.keras.layers.Activation('relu', name='fc7/relu')(x)
            if classes:
                x = tf.keras.layers.Dense(classes, name='fc8')(x)
                # x = tf.keras.layers.Activation('softmax', name='fc8/softmax')(x)        
        else:
            raise

    model = tf.keras.Model(input_, x, name='vggface_vgg16_block_%s' % block)
    if block != "top":
        weights_path = tf.keras.utils.get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                                VGG16_WEIGHTS_PATH_NO_TOP,
                                cache_subdir=VGGFACE_DIR)
    else:
        weights_path = tf.keras.utils.get_file('rcmalli_vggface_tf_top_vgg16.h5',
                                VGG16_WEIGHTS_PATH_TOP,
                                cache_subdir=VGGFACE_DIR)                                        
    model.load_weights(weights_path, by_name=True)
    return model