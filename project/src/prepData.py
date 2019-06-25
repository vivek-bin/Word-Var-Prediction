import numpy as np
import random
import json
from inspect import getsourcefile
from os.path import abspath
from os.path import dirname


from keras.utils.np_utils import to_categorical

PATH = dirname(dirname(dirname(abspath(getsourcefile(lambda:0))))) + "/"

PATH = PATH+"data\\"
JSON_PATH = PATH + "JSON\\"

CATEGORY_DICT = {}
CATEGORY_DICT["para"] = 0
CATEGORY_DICT["var"] = 1
CATEGORY_DICT["english"] = 2
CATEGORY_DICT["french"] = 3


CATEGORY_COUNT = len(CATEGORY_DICT.keys())
MAX_INPUT_SIZE = 200000

MAX_LENGTH = 32
MAX_FEATURES = 80		#updated after fetching actual encoding



def getData(inputFileName):
	f = open(PATH+inputFileName+".txt")
	l = [e.rstrip().lower() for e in f]
	f.close()
	
	random.shuffle(l)
	l = l[:MAX_INPUT_SIZE]
	
	tl = [(i,CATEGORY_DICT[inputFileName]) for i in l]
	
	return tl


def getAllData():
	data = []
	for fileName in CATEGORY_DICT.keys():
		data = data + getData(fileName)
		
	random.shuffle(data)

	features = [i for i,j in data]
	labels = [j for i,j in data]
	
	return features, labels


def getEncoding(features):
	distinctChar = []
	for rec in features:
		for chr in rec:
			if chr not in distinctChar:
				distinctChar.append(chr)
				
	distinctChar = sorted(distinctChar)
	
	charEncode = {ch: (index+1) for index,ch in enumerate(distinctChar)}
	
	MAX_FEATURES = len(charEncode)
	return charEncode
	
	
def saveDataEncoding():	
	features,labels = getAllData()
	encoding = getEncoding(features)
	
	
	featureJson = json.dumps(features)
	with open(JSON_PATH+"features.json", "w") as json_file:
		json_file.write(featureJson)

	labelJson = json.dumps(labels)
	with open(JSON_PATH+"labels.json", "w") as json_file:
		json_file.write(labelJson)
		
	encodeJson = json.dumps(encoding)
	with open(JSON_PATH+"encoding.json", "w") as json_file:
		json_file.write(encodeJson)
		
	
	
def vectorize(features,charEncode):
	vectFeatures = np.zeros((len(features), MAX_LENGTH))		#initialize zero array
	
	for i,rec in enumerate(features):
		for j,ch in enumerate(rec):
			try:
				vectFeatures[i][j] = charEncode[ch]
			except KeyError:
				print("Huh? " + ch)
		
	return vectFeatures
	
	
def loadVectorizedData():
	with open(JSON_PATH+"features.json", "r") as json_file:
		features = json.load(json_file)
		
	with open(JSON_PATH+"labels.json", "r") as json_file:
		labels = json.load(json_file)
		
	with open(JSON_PATH+"encoding.json", "r") as json_file:
		encoding = json.load(json_file)
	
	
	MAX_FEATURES = len(encoding)
	
	x = vectorize(features,encoding)
	y = to_categorical(labels)
		
	return x,y
	

if __name__ == "__main__":
	saveDataEncoding()
	
	