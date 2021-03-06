import json
import os

with open(os.path.join('..', 'data', 'weights_json.json'), 'r') as weights_file:
    weights = json.load(weights_file)
    ret = {}
    # ret[1] = weights['0']
    # ret[5] = weights['4']
    # ret[10] = weights['9']
    ret[20] = weights['20']
    print(len(ret))
    json.dump(ret, open(os.path.join("..", "data", "light_weights_json.json"),"w+"))