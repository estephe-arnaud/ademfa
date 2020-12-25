import tensorflow as tf

def VGG16(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          **kwargs):

  input_tensor *= 255
  input_tensor = tf.keras.applications.vgg16.preprocess_input(input_tensor)
  model = tf.keras.applications.vgg16.VGG16(
    include_top=include_top,
    weights=weights,
    input_tensor=input_tensor,
    input_shape=input_shape,
    pooling=pooling,
    classes=classes,
    **kwargs
  )

  return model
