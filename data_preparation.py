import numpy as np
import pandas as pd

import os

# Make Folders
outer_names = ["train", "test"]
inner_names = ["anger", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]

os.makedirs("data", exist_ok=True)

for outer_name in outer_names:
    os.makedirs(os.path.join("data", outer_name), exist_ok=True)

# Keep count of each category
category_track = { "anger": 0,
                   "contempt": 0,
                   "disgust": 0,
                   "fear": 0,
                   "happiness": 0,
                   "neutrality": 0,
                   "sadness": 0,
                   "surprise": 0
                 }
