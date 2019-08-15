from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from generator import text_image_generator
from model import get_model
from parameters import *
K.set_learning_phase(0)

model= get_model()
train_gen= text_image_generator(dir_path+"train")
train_gen.build_data()

test_gen= text_image_generator(dir_path+"test")
test_gen.build_data()

ada= Adadelta()
checkpoint= ModelCheckpoint(filepath='{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, period=1)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer= ada)
model.fit_generator(generator=train_gen.next_batch(),
                    steps_per_epoch=int(train_gen.n / batch_size),
                    epochs=30,
                    callbacks=[checkpoint],
                    validation_data=test_gen.next_batch(),
                    validation_steps=(test_gen.n // batch_size)) 
