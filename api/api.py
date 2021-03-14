from flask import Flask, jsonify, request, make_response, session

import argparse
import uuid
import json
import time
from tqdm import tqdm

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

from deepface import DeepFace

tf.config.set_visible_devices([], 'GPU')
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)

#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

#------------------------------

app = Flask(__name__)

## ngrok http 5000 --authtoken 1oWYEqQbxW2qxKFY8giP06RMzl4_58dAxo9nBzKP3aHZLvPTd

tic = time.time()

print("Loading Facial Attribute Analysis Models...")

pbar = tqdm(range(0,3), desc='Loading Facial Attribute Analysis Models...')

for index in pbar:
	if index == 0:
		pbar.set_description("Loading emotion analysis model")
		emotion_model = DeepFace.build_model('Emotion')
	elif index == 1:
		pbar.set_description("Loading age prediction model")
		age_model = DeepFace.build_model('Age')
	elif index == 2:
		pbar.set_description("Loading gender prediction model")
		gender_model = DeepFace.build_model('Gender')

toc = time.time()

facial_attribute_models = {}
facial_attribute_models["emotion"] = emotion_model
facial_attribute_models["age"] = age_model
facial_attribute_models["gender"] = gender_model

print("Facial attribute analysis models are built in ", toc-tic," seconds")

#------------------------------

if tf_version == 1:
	graph = tf.get_default_graph()

#------------------------------
#Service API Interface

@app.route('/')
def index():
	return '<h1>Hello, world!</h1>'

@app.route('/analyze', methods=['POST'])
def analyze():

	tic = time.time()
	req = request.get_json()
	trx_id = uuid.uuid4()

	#---------------------------
	resp_obj = analyzeWrapper(req, trx_id)
		
	#---------------------------

	toc = time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj, 200

def analyzeWrapper(req, trx_id = 0):
	resp_obj = jsonify({'success': False})

	instances = []
	if "img" in list(req.keys()):
		raw_content = req["img"] #list

		for item in raw_content: #item is in type of dict
			instances.append(item)
	
	if len(instances) == 0:
		return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205
	
	print("Analyzing ", len(instances)," instances")

	#---------------------------

	actions= ['emotion', 'age', 'gender']
	if "actions" in list(req.keys()):
		actions = req["actions"]
	
	#---------------------------

	resp_obj = DeepFace.analyze(instances, actions=actions, models=facial_attribute_models, enforce_detection = False, detector_backend = 'ssd' )
	
	return resp_obj
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--port',
		type=int,
		default=5000,
		help='Port of serving api')
	args = parser.parse_args()
	app.run(host='0.0.0.0', port=args.port)
