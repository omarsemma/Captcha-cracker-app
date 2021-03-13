import torch

# General  
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#MAP_LOCATION = "cuda:0" if torch.cuda.is_available() else None
PATTERN = '*' 

# App configs
ALLOWED_EXTENSIONS = ['png','jpeg','jpg']
IMAGES_UPLOADED_PATH = "./app/data/captcha_v1"
SECRET_KEY = r"__ca§!pt%^cha%@pp%"

# For training 
NUM_WORKERS = 8
BATCH_SIZE = 8
EPOCHS = 150

