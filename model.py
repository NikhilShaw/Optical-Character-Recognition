from keras.applications.resnet50 import ResNet50
from keras.models import Model, Sequential
from keras.layers import Input, Bidirectional, Dense, Reshape, Lambda, LSTM, BatchNormalization
from keras import backend as K
from parameters import *

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model():
	# defining the input layer, shape: (None, img_width, img_heigth, 1)
	inputs= Input(name='the_input', shape=(img_w, img_h, channel), dtype='float32')
	base_model = ResNet50(include_top=False, input_shape=(img_w, img_h, channel), weights='imagenet')(inputs)
	base_model.trainable = False
	reshape_layer= Reshape(target_shape=((32, 3136)), name='reshape')(base_model)
	fc1= Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(reshape_layer)
	#model = Sequential()
	# CNN to RNN
	#model.add(Reshape(target_shape=((32, )), name='reshape'))  # (None, 32, 2048)
	bi1= Bidirectional(LSTM(256, return_sequences=True))(fc1)
	bn1= BatchNormalization()(bi1)
	bi2= Bidirectional(LSTM(256, return_sequences=True))(bn1)
	bn2= BatchNormalization()(bi2)
	y_pred= Dense(num_classes, activation='softmax')(bn2)

	labels = Input(name='the_labels', shape=[max_text_len], dtype='float32') # (None ,8)
	input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
	label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)
	# Keras doesn't currently support loss funcs with extra parameters
	# so CTC loss is implemented in a lambda layer
	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1
	return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)