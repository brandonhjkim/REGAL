import pandas as pd
from ctgan import CTGAN 
from sdv.sampling import Condition 
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import models, transforms
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

# read in data
df = pd.read_csv('df_with_preds_and_embeddings.csv') # processed dataset with preds and embeddings

# preprocessing df
df.drop(columns='predicted_success', inplace=True) # don't need to include in ct-gan process
numeric_columns = df.select_dtypes(include='number').columns.tolist() 
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist() 
scaler = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns)
    ],
    remainder='passthrough'
)
df_scaled = scaler.fit_transform(df) 

# gpu init 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training ctgan 
ctgan = CTGAN(
    epochs = 500, 
    batch_size = 500,
    verbose = True, 
    device=device
)

ctgan.fit(df_scaled, categorical_columns=categorical_columns)

# loading in succsarnet

# from succsarnet.py 
class FusionModel(nn.Module):
    def __init__(self, tabular_input_dim):
        super().__init__()

        # resnet
        resnet = models.resnet50(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  
        self.cnn_out_dim = resnet.fc.in_features
        self.img_embedding = nn.Linear(self.cnn_out_dim, 128)

        # fusion mlp 
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + tabular_input_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=False), 
            nn.Linear(32, 1)  
        )

    def forward(self, image, tabular):
        img_features = self.cnn_backbone(image).reshape(image.size(0), -1)
        img_embeds = self.img_embedding(img_features)
        tab_embeds = self.tabular_mlp(tabular)
        combined = torch.cat((img_embeds, tab_embeds), dim=1)
        return self.fusion_mlp(combined).squeeze(1)

# reinit 
X_train = pd.read_csv('x_train.csv')
succsarnet = FusionModel(tabular_input_dim=len(X_train.columns))
succsarnet.load_state_dict(torch.load('succsarnet_model.pth', map_location=device)['model_state_dict'])
succsarnet.eval()

# running regal:

# generating realistic perturbations 
def generate_neighborhood(instance, success_bool, neighborhood_size, decision_boundary=0.5, perturbed_features_prop=0.5):
    random.seed(599) 

    # converting to dict
    instance_dict = instance.to_dict() 

    # randomly sampling actual values 
    p = max(1, int((1-perturbed_features_prop)*len(instance_dict)))

    neighborhood = pd.DataFrame() 
    y = np.array() 

    while len(neighborhood) < neighborhood_size:  
        # randomly selecting conditions for 1 perturbation
        sampled_keys = random.sample(list(instance_dict.keys()), p)
        condition = Condition(
            num_rows=1,
            column_values={k: instance_dict[k] for k in sampled_keys}
        )

        # generating from ctgan  
        synth = ctgan.sample_from_conditions(conditions=[condition])
        synth_numeric = torch.tensor(pd.get_dummies(synth).values, dtype=torch.float32).to(device) 

        # eval synthetic data
        with torch.no_grad(): 
            logit = succsarnet.fusion_mlp(synth_numeric)
            prob = torch.sigmoid(logit.squeeze(1)).cpu().numpy()
            
            # only keeping stuff within same decision boundary class 
            if (success_bool & prob >= decision_boundary) or (~success_bool & prob <= decision_boundary):
                neighborhood = pd.concat([neighborhood, synth])
                y = np.append(y, prob) 
    
    return neighborhood, y 

# obtaining importance and interplay  
def feature_interplay_and_importance(instance, neighborhood, y, min_impurity_decrease=0.0001):
    # one hot encoding 
    encoder = ColumnTransformer(
        transformer = [
            ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=object)),
            ], 
        remainder='passthrough')
    
    # to maintain consistency 
    instance.pop('predicted_success')
    full = encoder.fit_transform(pd.concat([instance, neighborhood])) 

    # removing original instance 
    updated_instance = full.iloc[0].to_numpy() 
    full = full.drop(full.index[0])

    # weighting 
    weights = cdist(full.values, updated_instance.reshape(1, -1), metric='euclidean').flatten()

    # decision tree fit 
    tree = DecisionTreeRegressor(min_impurity_decrease=min_impurity_decrease)
    tree.fit(full, y, sample_weight=weights)

    # importance
    X = np.array(full.values)
    n = len(y) 

    parent_variance = np.var(y)
    importance = {}

    for i in range(X.shape[1]):
        feature_values = X[:, i]
        # considering thresholds as all unique values 
        thresholds = np.unique(feature_values)
        best_reduction = -np.inf

        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            # computing IR 
            nL, nR = len(y_left), len(y_right)
            varL, varR = np.var(y_left), np.var(y_right)
            weighted_child_var = (nL / n) * varL + (nR / n) * varR
            reduction = parent_variance - weighted_child_var

            # only considers best split of a feature (naive approach) 
            if reduction > best_reduction:
                best_reduction = reduction

        key = full.columns[i]
        importance[key] = best_reduction

    return tree, importance 

# Grad-CAM

def gradcam_from_embedding(embed_index, image_tensor):
    activations = []
    gradients = []

    # hooks 
    def forward_hook(module, input, output):
        activations.append(output)
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # register hooks on the last resnet conv layer (layer4)
    hook_layer = succsarnet.cnn_backbone[-1]
    f_handle = hook_layer.register_forward_hook(forward_hook)
    b_handle = hook_layer.register_backward_hook(backward_hook)

    # ensure image is on same device and requires grad
    image_tensor = image_tensor.to(next(succsarnet.parameters()).device).requires_grad_(True)

    # forward pass through cnn + embedding
    img_features = succsarnet.cnn_backbone(image_tensor).reshape(1, -1)
    img_embeds = succsarnet.img_embedding(img_features)

    # define target
    target = img_embeds[0, embed_index]

    # backward pass
    succsarnet.zero_grad()
    target.backward()

    # activations and gradients from hooked layer
    acts = activations[0] 
    grads = gradients[0]   

    # grad-cam weights
    weights = grads.mean(dim=(2, 3), keepdim=True)
    grad_cam = F.relu((weights * acts).sum(dim=1, keepdim=True))

    # normalize
    heatmap = grad_cam.squeeze().detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # remove hooks 
    f_handle.remove()
    b_handle.remove()

    return heatmap

def visualize_gradcam_overlay(image_tensor, heatmap, alpha=0.5,):
    # convert to rgb 
    img = image_tensor.squeeze().detach().cpu().numpy()  
    img = np.transpose(img, (1, 2, 0)) 

    # resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # blend the heatmap and the image
    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * img)

    # visualize 
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')
    plt.show()

# example usage of visualizing grad_cam 

# preprocessing pipeline 
preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])   
    ])  

# load in image 
image = Image.open('MRALabeled2178.tif')
# preprocess 
image_tensor = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
# heatmap creation
heatmap = gradcam_from_embedding(embed_index=91, image_tensor=image_tensor)
# visualize
visualize_gradcam_overlay(image_tensor=image_tensor, heatmap=heatmap)


        

    

    




