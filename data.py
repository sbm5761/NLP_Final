import numpy as np
import pandas as pd
from utils import *
import math
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from load_images import ImageDataset
from models import *
import torch
from tqdm import tqdm
import cv2
import tensorflow as tf
import pickle
import os
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer

#Loading dataset
df = pd.read_csv('amazon_data.csv')

#Titles:
#UID, Product Name, Category, Selling Price, About Product, Product Specification, Technical Details, Product, Image, Product URL 

#Extracting only first N-samples
num_samples = 500
sample = df[:][:num_samples]
des= sample['Technical Details']
about= sample["About Product"]
id= sample["Uniq Id"]

#Get the images from the dataset and save into folder
#Only need to do once in running the model
def create_image():
    for index, row in sample.iterrows():
        url= row["Image"]
        UID= row["UID"]
        get_image(url, UID)

#Uses the image path and csv of descriptions to create and return a data_loader
def data_loader():
    image_paths = glob('./images/*.jpg')
    image_paths= image_paths[:num_samples]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])

    dataset = ImageDataset(image_paths, transform, captions_toked)

    return dataset

#get_desc: used to clean the dataset, only in my implementation
#cleaned_des= get_desc(des, about,id) #To populate excel sheet
cleaned_des = des

#Save and load tokenizer for dataset, need to do when running on different dataset sizes
if not os.path.exists("tokenizer.p"):
    token = tokenizer(cleaned_des)
else:
    token = pickle.load(open("tokenizer.p", "rb"))

#get the tokenized array for each description
captions_toked = get_tokens(token, cleaned_des)

#get the dataset
dataset = data_loader()
X_train, X_test = train_test_split(dataset, test_size=0.2) #split dataset

train_loader = DataLoader(X_train) 
test_loader = DataLoader(X_test) 

device = torch.device("mps")
vocab_size = len(token.word_index)

# Initialize the models. 
encoder = EncoderCNN(512)
decoder = DecoderRNN(512, 512, vocab_size)
encoder.to(device)
decoder.to(device)

#load in pre-trained model
encoder.load_state_dict(torch.load("encoder_500.pt"), strict=False)
decoder.load_state_dict(torch.load("decoder_500.pt"), strict=False)

#Loss Function
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.embed.parameters())
optimizer = torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-08)

def ROUGE_metric(predicted, original):
    breakpoint()
    scorer= rouge_scorer.RougeScorer(['rouge1']) #can change/experiment with various n-grams
    description = original.tolist()[0]
    desc_words = token.sequences_to_texts([description])
    score = scorer.score(predicted[0], desc_words[0])
    print("ROUGE Score:", score)

#To train the model
def train():
    print("Begin Training")

    #Train model
    epochs = 5
    for i in range(epochs):
        for indx, data in enumerate(tqdm(train_loader)):
            # Move batch of images and captions to GPU if CUDA is available.
            images = data[0].to(device)
            captions = data[1].long().to(device)

            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()

            feats= encoder(images)
            outputs = decoder(feats, captions)

            out= outputs.contiguous().view(-1, vocab_size)
            cap= captions.view(-1)  
            loss = criterion(out, cap)
            # Backward pass.
            loss.backward()
            
            # Update the parameters in the optimizer.
            optimizer.step()
    
    #Save Model
    #torch.save(encoder.state_dict(), "encoder_500.pt")
    #torch.save(decoder.state_dict(), "decoder_500.pt")

#To test the model
def test():
    print("Begin Testing")
    #Test Model
    encoder.eval()
    decoder.eval()

    for indx, data in enumerate(test_loader):
        image= data[0]
        cap = data[1] 

        image= image.to(device)
        cap= cap.to(device)

        features= encoder(image).unsqueeze(1)
        output= decoder.sample(features)  

        predicted_caption= token.sequences_to_texts([output])
        print(predicted_caption) 

        ROUGE_metric(predicted_caption, torch.tensor(cap, dtype=torch.int32)) #Get ROUGE metric for predicted value


train()
test()