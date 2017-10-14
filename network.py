############################################################
# ----- Imports -----
############################################################
import numpy 
from keras.datasets             import cifar10 
from keras.models               import Sequential 
from keras.layers               import Dense 
from keras.layers               import Dropout 
from keras.layers               import Flatten 
from keras.constraints          import maxnorm 
from keras.optimizers           import SGD 
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D 
from keras.utils                import np_utils 
from keras                      import backend as k
k.set_image_dim_ordering('th')

############################################################
# ----- Fix Seed -----
############################################################
seed = 7 
numpy.random.seed(seed)


############################################################
# ----- Load Data -----
############################################################
(X_train, y_train) , (X_test, y_test) = cifar10.load_data()

# Extra step where I select only part of the data due to filling the memory
# Notes: 
# 1. Very low sample size introduces weird error, not sure why 
# 2. A bit larger in range 30, gives accuracy = 0. This is logical, as training data if very small, the nework learnt nothing :)  
training_sample_size = 100
testing_sample_size  = 20

X_train = X_train[0:training_sample_size]
y_train = y_train[0:training_sample_size]

X_test = X_test[0:testing_sample_size]
y_test = y_test[0:testing_sample_size]

############################################################
# ----- Normalize Inputs -----
############################################################
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32') 

X_train = X_train/255.0 
X_test  = X_test/255.0 

############################################################
# ----- Hot One Encoding -----
############################################################
y_train = np_utils.to_categorical(y_train) 
y_test  = np_utils.to_categorical(y_test) 
num_classes = y_train.shape[1]
print('num_classes = {}'.format(num_classes))
############################################################
# ----- Create a Model -----
############################################################
model = Sequential() 
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode= 'same' ,
  activation= 'relu' , W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation= 'relu' , border_mode= 'same' ,
W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation= 'relu' , W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation= 'softmax' ))

############################################################
# ----- Compile Model -----
############################################################
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss= 'categorical_crossentropy' , optimizer=sgd, metrics=[ 'accuracy' ])
print(model.summary())

############################################################
# ----- Fit Model -----
############################################################
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs,
batch_size=32, verbose=2)

############################################################
# ----- Final Evaluation of Model -----
############################################################
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



