import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

import prepData as PD


TRAIN_SPLIT = 500000

PATH = PD.PATH
MODEL_PATH = PATH + "models\\"


def getData():
	x,y = PD.loadVectorizedData()
		
	xTrain = x[:TRAIN_SPLIT]
	xTest = x[TRAIN_SPLIT:]
	yTrain = y[:TRAIN_SPLIT]
	yTest = y[TRAIN_SPLIT:]
	
	return xTrain,yTrain,xTest,yTest


def trainConvModel(xTrain,yTrain):
	model = Sequential()
	model.add(layers.Embedding(PD.MAX_FEATURES, 4, input_length=PD.MAX_LENGTH))
	model.add(layers.Conv1D(8, 7, activation='relu'))
	model.add(layers.MaxPooling1D(3))
	model.add(layers.Conv1D(4, 5, activation='relu'))
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(PD.CATEGORY_COUNT, activation='softmax'))
	#model.summary()
	model.compile(	optimizer=RMSprop(lr=8e-4),
					loss='categorical_crossentropy',
					metrics=['acc'])
	
	
	history = model.fit(xTrain, yTrain,
					epochs=40,
					batch_size=128,
					validation_split=0.2)
	
	model.summary()
	saveModel(model, "label_model")
	
	return model
	

def trainTryModel(xTrain,yTrain):
	model = Sequential()
	model.add(layers.Embedding(PD.MAX_FEATURES, 64, input_length=128))
	model.add(layers.LSTM(32,return_sequences=True))
	model.add(layers.LSTM(32))
	model.compile(	optimizer=RMSprop(lr=8e-4),
					loss='categorical_crossentropy',
					metrics=['acc'])
	
	model.summary()
	

def saveModel(model,modelName):
	# serialize model to JSON
	model_json = model.to_json()
	with open(MODEL_PATH + modelName + ".json", "w") as json_file:
		json_file.write(model_json)
	
	# serialize weights to HDF5
	model.save_weights(MODEL_PATH + modelName + ".h5")
	print("Saved model to disk")
	

	
if __name__ == "__main__":
	xTrain,yTrain,xTest,yTest = getData()
	#model = trainConvModel(xTrain,yTrain)
	trainTryModel(xTrain,yTrain)
	#scores = model.evaluate(xTest, yTest, verbose=0)
	#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


