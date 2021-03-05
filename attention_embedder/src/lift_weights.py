import json
import os

with open('../data/weights_gz.json', 'r') as weights_file:
    weights = json.load(weights_file)
    ret = {}
    ret[1] = weights['0']
    ret[5] = weights['4']
    ret[10] = weights['9']
    ret[20] = weights['19']
    json.dump(ret, open(os.path.join("..", "data", "light_weights_gz.json"),"w+"))
