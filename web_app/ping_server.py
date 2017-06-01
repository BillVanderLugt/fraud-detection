import requests
import json
import pandas as pd
from build_model import Model, get_data
import pickle

def ping():
    """
    Makes request to server hosting rotating data points. Parses the json into features appropriate for our random forest to predict on. Returns tuple containing relevant features for ML model.
    """
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    resp = requests.get(url)
    json_data = json.loads(resp.text)

    body_length = json_data['body_length']
    sale_duration2 = json_data['sale_duration2']
    user_age = json_data['user_age']
    name_length = json_data['name_length']
    payee_ind = (len(json_data['payee_name']) > 0) * 1
    user_type = json_data['user_type']
    fb_published = json_data['fb_published']

    record = (body_length, sale_duration2, user_age, name_length, payee_ind, user_type, fb_published)

    return record
