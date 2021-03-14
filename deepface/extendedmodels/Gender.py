from deepface.basemodels import VGGFace
import os
from pathlib import Path
import gdown
import numpy as np

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	from keras.models import Model, Sequential
	from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Convolution2D, Flatten, Activation, MaxPooling2D, Dense

def loadModel( version = '2'):
	
	#--------------------------
	if version == '1' :
		classes = 2

		model = VGGFace.baseModel()

		base_model_output = Sequential()
		base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
		base_model_output = Flatten()(base_model_output)
		base_model_output = Activation('softmax')(base_model_output)
		
		#--------------------------

		gender_model = Model(inputs=model.input, outputs=base_model_output)
	
		#load weights
	
		home = str(Path.home())
		
		if os.path.isfile(home+'/.deepface/weights/gender_model_weights.h5') != True:
			print("gender_model_weights.h5 will be downloaded...")
			
			url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'
			output = home+'/.deepface/weights/gender_model_weights.h5'
			gdown.download(url, output, quiet=False)
		
		gender_model.load_weights(home+'/.deepface/weights/gender_model_weights.h5')
		
	#--------------------------
	if version == '2' :
		classifier = Sequential()

		classifier.add(Convolution2D(32, 3, input_shape = train_data[0].shape, activation = 'relu'))
		classifier.add(MaxPooling2D(pool_size = (2,2)))

		classifier.add(Convolution2D(64, 3, activation = 'relu'))
		classifier.add(MaxPooling2D(pool_size = (2,2)))

		classifier.add(Flatten())
		# classifier.add(Dropout(0.2))
		classifier.add(Dense(units = 512, activation = 'relu'))
		classifier.add(Dense(units = 1, activation = 'sigmoid'))

		gender_model = Model(inputs=model.input, outputs=base_model_output)

        home = str(Path.home())
        gender_model.load_weights(home+'/.deepface/weights/trained_cnn.h5')

	return gender_model
	
	#--------------------------