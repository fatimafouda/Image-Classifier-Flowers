import numpy as np
import torch
from torch import nn, optim
from PIL import Image
from torchvision import datasets, transforms, models

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    accuracy = checkpoint['accuracy']
    epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']
    model_type = checkpoint['model_type']
    print("Pretrained model type: ",model_type,)
    print("Epochs: ", epoch,)
    print("Accuracy: ",round(accuracy*100,1),"%")
    return model

def preproc_image(filepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
    '''
    #Open image
    im = Image.open(filepath)
    #Use transforms for faster processing
    process = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = process(im)
    return img_tensor

def predict(image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    image = image.unsqueeze(0)
    image.to(device)
    model = model.cuda()
    
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        top_prob = top_prob.exp()
        top_prob_array = top_prob.data.numpy()[0]
    
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    
    top_labels_data = top_labels.data.numpy()
    top_labels_list = top_labels_data[0].tolist()  
    
    top_classes = [inv_class_to_idx[x] for x in top_labels_list]
    print("top probability: ",top_prob_array)
    print("top class: ",top_classes)
    return top_prob_array, top_classes
        
def flower_type(label, cats):
   print([cats[x] for x in label])
