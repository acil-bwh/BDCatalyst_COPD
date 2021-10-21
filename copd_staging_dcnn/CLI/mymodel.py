from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def build_model(image_dims, n_slices, weightsPath=None):
    # First, build the 2D CNN model:
    ipt = Input(shape=image_dims, name='CNN_Input')
    x = Conv2D(32, (6, 6), activation='relu', name='Conv2D_1')(ipt)
    x = MaxPooling2D(pool_size=(3, 3), name='MaxPooling2D_1')(x)
    x = Conv2D(96, (3, 3), activation='relu', name='Conv2D_2')(x)
    x = MaxPooling2D(pool_size=(3, 3), name='MaxPooling2D_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='Conv2D_3')(x)
    x = MaxPooling2D(pool_size=(4, 4), name='MaxPooling2D_3')(x)
    x = Conv2D(384, (3, 3), activation='relu', name='Conv2D_4')(x)
    x = MaxPooling2D(pool_size=(4, 4), name='MaxPooling2D_4')(x)
    x = Flatten(name='Flatten')(x)

    cnn = Model(inputs=ipt, outputs=x, name='cnn')

    # Then, wrap the whole CNN into a LSTM framework:
    lstm_input_shape = (n_slices,) + image_dims
    lstm_in = Input(shape=lstm_input_shape, name='Input')
    x = TimeDistributed(cnn, name='TimeDistributedFeatures')(lstm_in)
    x = LSTM(1228, name='LSTM')(x)
    x = Dropout(0.5, name='Dropout')(x)
    x = Dense(1024, activation='relu', name='Dense_1')(x)
    opt = Dense(5, activation='softmax', name='Output')(x)

    final_model = Model(inputs=lstm_in, outputs=opt, name='CNN-LSTM')

    if weightsPath:
        final_model.load_weights(weightsPath)

    return final_model
