#IMPORT LIBRARIES
from keras.preprocessing.image import ImageDataGenerator   #FOR DATA AUGMENTATION
from keras.models import Sequential   #FOR THE MODEL TO BE TRAINED
from keras.layers import Conv2D, MaxPooling2D   #FOR CONVOLUTIONAL LAYERS
from keras.layers import Activation, Dropout, Flatten, Dense   #FOR CONVOLUTIONAL LAYERS
from keras import backend as K   #FOR KERAS BACKEND PROCESSES
train_data_dir = r'C:\Users\aydin\Desktop\svm\dataset\train'   #FILE PATH OF THE IMAGES TO BE GIVEN TO THE MODEL FOR TRAIN
validation_data_dir = r'C:\Users\aydin\Desktop\svm\dataset\valid'   #FILE PATH OF THE IMAGES TO BE GIVEN TO THE MODEL FOR VALIDATION
nb_train_samples = 720   #NUMBER OF TRAIN IMAGES
nb_validation_samples = 120   #NUMBER OF VALIDATION IMAGES
epochs = 1   #TRAINING NUMBER FOR MODEL
batch_size = 20   #DURING TRAINING, THE MODEL WORKS WITH GROUPS OF BATCH_SIZE IMAGES.
img_width, img_height = 224, 224   #SIZE OF TRAIN AND VALIDATION IMAGES
#ACCORDING TO DATA FORMAT, INPUT_SHAPE IS DETERMINED. (3, 224, 244) OR (22,4, 224, 3)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
#THE MODEL IS DESIGNED IN THIS PART.
model = Sequential()   #A SEQUENTIAL MODEL IS CREATED.
model.add(Conv2D(32, (3, 3), input_shape=input_shape)) #A CONVOLUTIONAL LAYER WITH 32 NEUROS IS CREATED. KERNEL SIZE 3X3 AND INPUT_SHAPE IS INDICATED.
model.add(Activation('relu'))   #THE ACTIVATION FUNCTION IS RELU.
model.add(MaxPooling2D(pool_size=(2, 2)))   #MAX_POOLING 2X2 IS ADDED TO CONVOLUTIONAL LAYER.
model.add(Conv2D(32, (3, 3), input_shape=input_shape)) #A CONVOLUTIONAL LAYER WITH 32 NEUROS IS CREATED. KERNEL SIZE 3X3 AND INPUT_SHAPE IS INDICATED.
model.add(Activation('relu'))   #THE ACTIVATION FUNCTION IS RELU.
model.add(MaxPooling2D(pool_size=(2, 2)))   #MAX_POOLING 2X2 IS ADDED TO CONVOLUTIONAL LAYER.
model.add(Dropout(0.25))   #THE DROPOUT PROCESS IS IMPLEMENTED
model.add(Flatten())   #FLATTENING TWO FULLY CONNECTED LAYERS
model.add(Dense(512))   #ADDING TWO FULLY CONNECTED LAYERS:
model.add(Activation('relu'))   #THE ACTIVATION FUNCTION IS RELU.
model.add(Dropout(rate=0.5))   #THE DROPOUT PROCESS IS IMPLEMENTED
model.add(Dense(1))   #MAKE FULLY CONNECTED MODEL
model.add(Activation('sigmoid'))   #THE ACTIVATION FUNCTION IS SIGMOID FOR OUTPUT LAYER.
model.compile(loss='binary_crossentropy',   #MODEL COMPILED
              optimizer='adam',  #ADAM IS USED FOR OPTIMIZATION
              metrics=['accuracy'])   #FIND MODEL ACCURACY
train_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
        height_shift_range=0.2, rescale=1./255, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest') #DATA AUGMENTATION FOR TRAIN IMAGES
test_datagen = ImageDataGenerator(rescale=1./255)   #DATA AUGMENTATION FOR VALIDATION IMAGES
train_generator = train_datagen.flow_from_directory(train_data_dir,                                                  target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary') #TRAIN DATA PREPARE FOR MODEL
validation_generator = test_datagen.flow_from_directory( validation_data_dir,
 target_size=(img_width, img_height),
 batch_size=batch_size, class_mode='binary') #VALIDATION DATA PREPARE FOR MODEL
    history = model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs, validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size)
print(model.summary())   #PRINT MODEL SUMMARY
model.save("norm-relu-adam15.h5")   #SAVE MODEL TRAINED
