import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import json
import os
#from pathlib import Path
from collections import OrderedDict



parser = argparse.ArgumentParser()
parser.add_argument('--img_path', action='store', dest='img_path', help='Path of image to predict', required=True)
parser.add_argument('--chk_path', action='store', dest='checkpoint_path', help='Path of checkpoint, has to finish with /', required=True)
parser.add_argument('--top_k', action="store", dest="top_k", default=5,  type=int, help="Top names to compare an image")
parser.add_argument('--name_flowers', action="store",dest="name_flowers",  default='cat_to_name.json',  help="The json with the category and names")
parser.add_argument('--gpu', action='store',dest='gpu', default=True,  help='Use GPU')

results = parser.parse_args()
img_path = results.img_path
top_k = results.top_k
name_flowers = results.name_flowers
gpu = results.gpu

save_dir=results.checkpoint_path #'checkpoints/'

#add the / at the end
if(save_dir[-1]!='/'):
	save_dir=save_dir+'/'
	
	
if(os.path.exists(save_dir)):
	save_dir=save_dir+"checkpoint.pth"
else:
    raise Exception("Directory for saved model %s doesn't exists" % save_dir)


if gpu==True:
    using_gpu = torch.cuda.is_available()
    processor='gpu'
else:
    processor='cpu'
    
print('Using '+processor)


#

def define_model(hidden_sizes,output_size,architecture):
    
    if architecture=="vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
    else:
        model = models.mobilenet_v2(pretrained=True)
        input_size = 2208
    
    for param in model.parameters():
        param.requires_grad = False  
        
    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(0.5)),
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_sizes[1], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    

    model.classifier=classifier
    return model


  
def load():
    state_model = torch.load(save_dir)
    learning_rate=state_model['learning_rate']
    epochs=state_model['epochs']
    hidden_sizes=state_model['hidden_sizes']
    output_size=state_model['output_size']
    arch=state_model['arch']

    model=define_model(hidden_sizes,output_size,arch)
    model = model.cuda()

    model.class_to_idx=state_model['class_to_idx']

    model.load_state_dict(state_model['state_dict'])
        
    
    print('model load')
    
    return model

def process_image():
    
    img = Image.open(img_path)
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(img)
    print('image processed')

    return img

def predict():
 
    model.eval()
    model.cpu()
    
    img = process_image()
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img)
        probs, classes = torch.topk(output,top_k)
        probs = probs.exp()


    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_n = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
        
    return probs, top_n


def load_Names():
      with open(name_flowers, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

def get_Flower_Name(probability,classes):
	
	labels = []
	for index in classes:
		labels.append(cat_to_name[str(index)])

	print('Name of the given image: ', labels[0])

	probs=probability[0]

	for name, prob in zip(labels, probs):
		print("Name of class and probability {}: {:6f}".format(name, prob))



# Label mapping
cat_to_name = load_Names()

#load model
model=load()

#predict image
probs, classes = predict()

#get name flower and top probabilitys
get_Flower_Name(probs, classes)



#example call
#python predict.py --img_path "flowers/test/74/image_01191.jpg" --chk_path "checkpoints/"








