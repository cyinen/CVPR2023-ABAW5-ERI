import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#3090
# adjust your paths here. Recommended to keep it that way in order not to run into git conflicts
BASE_PATH = '/data/ABAW2022/dataset/'
OUTPUT_PATH = '/data/ABAW2022/dataset/'

# dev
# BASE_PATH = '/chenyin2/project/dataset/'
# OUTPUT_PATH = '/chenyin2/project/code/'


PATH_TO_FEATURES = {
    'reaction': os.path.join(BASE_PATH, 'Hume/features'),
}

# humor is labelled every 2s, but features are extracted every 500ms
N_TO_1_TASKS = {'reaction'}

ACTIVATION_FUNCTIONS = {
    'reaction': torch.nn.Sigmoid,
}

NUM_TARGETS = {
    'reaction': 7,
}


PATH_TO_LABELS = {
    'reaction': os.path.join(BASE_PATH, 'Hume'),
}

PATH_TO_METADATA = {
    'reaction':os.path.join(BASE_PATH, 'Hume'),
}

PARTITION_FILES = {task: os.path.join(path_to_meta, 'partition_train_val.csv') for task,path_to_meta in PATH_TO_METADATA.items()}

REACTION_LABELS = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']

OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'results')
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction')

FEATURE_LENGTH = {  'DeepSpectrum':1024,
                    'PosterV2+Vit':768*2}