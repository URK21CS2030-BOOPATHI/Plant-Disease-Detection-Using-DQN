import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# Define the CNN model architecture (same as the training script)
class CNN_NeuralNet(torch.nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = self.ConvBlock(in_channels, 64)
        self.conv2 = self.ConvBlock(64, 128, pool=True)
        self.res1 = torch.nn.Sequential(self.ConvBlock(128, 128), self.ConvBlock(128, 128))
        self.conv3 = self.ConvBlock(128, 256, pool=True)
        self.conv4 = self.ConvBlock(256, 512, pool=True)
        self.res2 = torch.nn.Sequential(self.ConvBlock(512, 512), self.ConvBlock(512, 512))
        self.classifier = torch.nn.Sequential(
            torch.nn.MaxPool2d(4),
            torch.nn.Flatten(),
            torch.nn.Linear(512, num_diseases)
        )

    def ConvBlock(self, in_channels, out_channels, pool=False):
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  torch.nn.BatchNorm2d(out_channels),
                  torch.nn.ReLU(inplace=True)]
        if pool:
            layers.append(torch.nn.MaxPool2d(4))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = CNN_NeuralNet(3, 38)
    model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Define a function to make predictions
def predict_image(image, model):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Streamlit app layout
st.title("Plant Disease Detection 🍃")
st.write("Upload an image of a plant leaf to detect diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
class_names = ['Tomato___Late_blight',
 'Tomato___healthy',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Potato___healthy',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Tomato___Early_blight',
 'Tomato___Septoria_leaf_spot',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Strawberry___Leaf_scorch',
 'Peach___healthy',
 'Apple___Apple_scab',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Bacterial_spot',
 'Apple___Black_rot',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Peach___Bacterial_spot',
 'Apple___Cedar_apple_rust',
 'Tomato___Target_Spot',
 'Pepper,_bell___healthy',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Potato___Late_blight',
 'Tomato___Tomato_mosaic_virus',
 'Strawberry___healthy',
 'Apple___healthy',
 'Grape___Black_rot',
 'Potato___Early_blight',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Common_rust_',
 'Grape___Esca_(Black_Measles)',
 'Raspberry___healthy',
 'Tomato___Leaf_Mold',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Pepper,_bell___Bacterial_spot',
 'Corn_(maize)___healthy']
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Classifying...")
    class_idx = predict_image(image, model)
    # class_names = os.listdir('train_directory_path')  # Replace with your classes
    st.write(f"Prediction: {class_names[class_idx]}")
