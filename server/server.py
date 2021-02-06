import sys
sys.path.append('../')
from flask import Flask, make_response, request
from get_features.features.extractor import Extractor
from elm_model import ExtremeLearningMachine
import math
import numpy as np
import time
app = Flask(__name__)

def extract_features(domain):
    features = []
    ext = Extractor(domain,name = True, web = True, dns=True)
    features.append(ext.get_length())
    features.append(ext.get_n_ns())
    features.append(ext.get_n_constant_chars())
    features.append(ext.get_n_vowel_chars())
    features.append(ext.get_life_time())
    features.append(ext.get_num_ratio())
    features.append(ext.get_n_labels())
    #features.append(ext.get_mean_TTL())
   # features.append(ext.get_n_constants())
    #features.append(ext.get_n_mx())
   # features.append(ext.get_vowel_ratio())
    #features.append(ext.get_n_nums())
#    features.append(ext.get_entropy())
#    features.append(ext.get_stdev_TTL())
#    features.append(ext.get_n_ip())
#    features.append(ext.get_active_time())
   # features.append(ext.get_n_vowels())
   # features.append(ext.get_rv())
   # features.append(ext.get_ns_similarity())
   # features.append(ext.get_vowel_constant_convs())
   # features.append(ext.get_alpha_numer_convs())
   # features.append(ext.get_n_other_chars())
#    features.append(ext.get_max_consecutive_chars())
   # features.append(ext.get_n_ptr())
#    features.append(ext.get_n_countries())
    return features


def load_model(shi_work,n_unit=600):
    model = ExtremeLearningMachine(n_unit=n_unit)
    if shi_work:
        model.load_weights("model/elm_shi.npz")
    else:
        model.load_weights("model/elm_threshold{}_nunit{}.npz".format(th,n_unit))
    return model


def normalize(data):
    param = np.load("param/param_threshold{}_nunit600.npz".format(th))
    mean,var = param["mean"],param["var"]
    data = np.array([(x - m)/math.sqrt(v) for x,m,v in zip(data,mean,var)])
    return data.reshape(1,data.shape[0])


def inference(model,data):
    result = model.predict(data)
    print(result)
    return result


@app.route("/", methods=['GET', 'POST'])
def process():
    if request.method == 'GET':
        return 'OK'
    elif request.method == 'POST':
        start_time = time.time()
        data = request.get_data().decode('utf-8')
        features = extract_features(data)
        print("extraction time = ",time.time()-start_time)
        start_inference = time.time()
        print(features)
        features = normalize(features)
        result = inference(model,features)
        print("inference_time = ",time.time()-start_inference)
        resp = make_response(str(result[0]))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
th = 7
n_unit=600
shi_work=False
model = load_model(shi_work,n_unit=n_unit)        
app.run(host='0.0.0.0', port=4000, threaded=True)
