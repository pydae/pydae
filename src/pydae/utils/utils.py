import json
import os

def read_data(data_input=''):
    
    if type(data_input) == str:
        if 'http' in data_input:
            url = data_input
            resp = requests.get(url)
            data = json.loads(resp.text)
        else:
            if os.path.splitext(data_input)[1] == '.json':
                with open(data_input,'r') as fobj:
                    data = json.loads(fobj.read().replace("'",'"'))
            if os.path.splitext(data_input)[1] == '.hjson':
                import hjson
                with open(data_input,'r') as fobj:
                    data = hjson.loads(fobj.read().replace("'",'"'))
    elif type(data_input) == dict:
        data = data_input

    return data