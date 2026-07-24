# necessary for API usage

# can delete these if not running locally
from . import model
from .basic import classify
from . import basic
import time
import os
import requests
import urllib.parse
# api classification defined here
def classify_api(sentence):
    sentence = urllib.parse.quote(sentence)
    response=requests.get('http://app.chat314.com/api/'+sentence)
    response=response.json()
    output_tag=response['output']
    prob_int=response['certainty']
    return output_tag, float(prob_int)
# init function defined
def init(location,key):
    x, y, z =s_init(location,key)
    return x, y, z
# download packages if necessary
def download():
    print('Downloading the model from GitHub. Press Ctrl+c to quit.')
    time.sleep(3)
    os.system('wget https://github.com/eedeb/Classy/raw/main/train/data.pth')
