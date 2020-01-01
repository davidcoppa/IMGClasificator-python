#imports
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import os


# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--path', action='store', dest='path', help='Path of directory with images', required=True)
parser.add_argument('--save_dir', action='store', dest='chk_path', default='checkpoints/', help='Path of checkpoint, has to finish with /')
parser.add_argument('--learning_rate', action='store',dest='learning_rate', default=0.001, type=float,  help='Learning rate of the network')
parser.add_argument('--epochs', action='store', dest='epochs', default=4, type=int, help='Number of epochs in the training')
parser.add_argument('--hidden_sizes', action='store', dest='hidden_sizes',default=[10200, 1020], nargs=2, type=int, help='2 hidden values for the network')
parser.add_argument('--gpu', action='store', dest='gpu', default=True,  help='Use GPU')
parser.add_argument('--arch', action='store', dest='arch', default='vgg16', help='Set the training architecture')

results = parser.parse_args()


#set values
data_dir = results.path
checkpoint_path = results.chk_path
hidden_sizes = results.hidden_sizes
#print(hidden_sizes[0])
#print(hidden_sizes[1])

epochs = results.epochs
learning_rate = results.learning_rate
gpu = results.gpu
arch=results.arch
print_every = 10
output_size=102 #total flowers

if gpu==True:
    using_gpu = torch.cuda.is_available()
    processor='gpu'
else:
    processor='cpu'
    
print('using '+processor)

#load directories
train_dir = data_dir + '/train/'
test_dir = data_dir + '/test'
valid_dir = data_dir + '/valid'


#add the / at the end
if(checkpoint_path[-1]!='/'):
	checkpoint_path=checkpoint_path+'/'
	
save_dir=checkpoint_path

#validate chk point folder
if(checkpoint_path!='checkpoints/'):
	save_dir=checkpoint_path


#validation checkpoint directory (bether now than after the training)
if(os.path.exists(save_dir)):
    print ("Directory for save the model %s already exists" % save_dir)
    save_dir=save_dir+'checkpoint.pth'
    
else:
    try:
        os.mkdir(save_dir)
    except OSError:
        print ("Creation of the directory %s failed, we use the root directory" % save_dir)
        save_dir='checkpoint.pth'
    else:
        print ("Successfully created the directory %s for save the model" % save_dir)
        save_dir=save_dir+'checkpoint.pth'


#load datasets

def load_datasets():
	#dataloaders

	data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
	image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
	image_data = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)

    #test
	testval_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
	image_testset = datasets.ImageFolder(test_dir, transform=testval_transforms)
	image_test = torch.utils.data.DataLoader(image_testset, batch_size=64, shuffle=True)
	
	#val
	val_transforms = transforms.Compose([transforms.Resize(256),
										transforms.CenterCrop(224),
										transforms.ToTensor(),
										transforms.Normalize([0.485, 0.456, 0.406],
															 [0.229, 0.224, 0.225])])

	image_valset = datasets.ImageFolder(valid_dir, transform=val_transforms)
	image_val = torch.utils.data.DataLoader(image_valset, batch_size=64, shuffle=True)

	return image_data,image_test,image_val,image_datasets

def define_model():
    
    if arch=="vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088 
        print("using architecture: vgg16")

    else:
        model = models.mobilenet_v2(pretrained=True)
        print("using architecture: mobilenet_v2")
        input_size = 1280
    
    #freeze parameters - less memory used
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

def get_accuracy(mod, data):
	loss = 0
	accuracy = 0
	data_len=len(data)
	for i, (inputs,labels) in enumerate(data):
                inputs, labels = inputs.to('cuda') , labels.to('cuda')
                mod.to('cuda')
                with torch.no_grad():    
                    outputs = mod.forward(inputs)
                    loss = criterion(outputs,labels)
                    ps = torch.exp(outputs).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
	loss = loss / data_len
	accuracy = accuracy /data_len
    
	return loss, accuracy

def training():
    print('start training')
    model.to('cuda')
    #model.share_memory()
    step=0

    for epo in range(epochs):
        running_loss=0

        for i,(inputs,labels) in enumerate(image_data_load):
            step+=1
            #print(step)
			
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs=model.forward(inputs)
            
            loss= criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            
            if step % print_every == 0:
                model.eval()
                val_loss,accuracy=get_accuracy(model,image_val_load)
               
                print("Step nro: {}  ".format(step),
                      "Epoch: {}/{}  ".format(1+epo, epochs),
                      "Loss: {:.4f}  ".format(running_loss),
                      "Validation Loss {:.4f}  ".format(val_loss),
                      "Accuracy {:.4f}  ".format(accuracy))

                running_loss = 0
    
    print('end training')

def testing():
	correctos = 0
	total = 0
	
	model.eval()
	model.to('cuda')
	
	with torch.no_grad():
		for inputs, labels in image_test_load:
			inputs, labels = inputs.to('cuda'), labels.to('cuda')
			outputs = model(inputs)
			aux , prediction = torch.max(outputs.data, 1)
			total += labels.size(0)
			tensor= (prediction == labels.data).sum()
			correctos+=tensor.item()
			
	
	accuracy=100 * correctos / total
	
	print('Total: {} - Correct: {} - Accuracy: {:.2f}% '.format(total,correctos,accuracy))


def save():
    
    model.class_to_idx = image_datasets.class_to_idx
    model_state={
        'learning_rate':learning_rate,
        'epochs':epochs,
        'hidden_sizes':hidden_sizes,
        'output_size':output_size,
        'state_dict':model.state_dict(),    
        'class_to_idx':model.class_to_idx,
        'arch':arch
    }

    torch.save(model_state, save_dir) 
    print('model saved')
      
        
#load dataset
image_data_load, image_test_load, image_val_load, image_datasets = load_datasets()

#load network

#training 
model=define_model()

model = model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

#training network
#torch.cuda.empty_cache()
training()

#testing network
testing()

#save network
save()


#example call
#python train.py --path flowers --save_dir checkpoints/
#python train.py --path flowers --save_dir checkpoints/ --arch mobilenet_v2
#python train.py --path flowers --save_dir checkpoints/ --hidden_sizes 5000 1020











