# necessary for API usage
from .functions import personal
from .functions import chat_gpt
from .functions import dall_e
from .functions import search
# can delete these if not running locally
from .server import server
from . import model
from .basic import classify
from .server import init as s_init
from . import basic
import time
import os
import nltk
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
try:
    import nltk
    nltk.word_tokenize('')
except LookupError:
    print('Please download the nltk punkt tokenizer by running this:')
    print("nltk.download('punkt')")
    print('If the error persists, run this:')
    print("nltk.download('punkt_tab')")
    print('Or try moving the tokenizer file')
