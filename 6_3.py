#Based on and some code utilized from link: https://pytorch.org/vision/stable/models.html

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import os

#Validation
data_dir = "C:/Users/prana/Desktop/Carnegie Mellon/Computer Vision/16720_S23_hw5/6_3 images/"

match_count = 0
file_count = 0
for filename in os.listdir(data_dir):
    try:
        file_count += 1
        img = read_image(data_dir + filename)

        # Step 1: Initialize model with the best available weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()

        # Step 2: Initialize the inference transforms
        preprocess = weights.transforms()

        # Step 3: Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)

        # Step 4: Use the model and print the predicted category
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]

        if category_name == "water bottle":
            match_count += 1
        
    except:
        #Ignoring black and white images as they do not work on the model
        print(filename + " failed")

valid_acc = match_count/file_count * 100
print("Validation Accuracy:", valid_acc)
