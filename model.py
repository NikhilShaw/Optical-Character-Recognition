from keras.applications.resnet import ResNet50
from keras.models import Model, Sequential
from keras.layers import Bidirectional, Dense, Reshape, Lambda

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


num_classes=10
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
model = Sequential()
model.add(base_model)
# CNN to RNN
model.add(Reshape(target_shape=((32, 2048)), name='reshape'))  # (None, 32, 2048)
model.add(Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')) # (None, 32, 64)
model.add(Bidirectional(256,  merge_mode='concat'))
model.add(Bidirectional(256,  merge_mode='concat'))
model.add(Dense(num_classes, activation='softmax'))

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)




