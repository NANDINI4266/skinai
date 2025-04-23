import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SkinDiseaseClassifier(nn.Module):
    """
    A neural network model for skin disease classification using a pre-trained ResNet50 backbone.
    """
    
    def __init__(self, num_classes=5):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of disease classes to predict
        """
        super(SkinDiseaseClassifier, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.base_model = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the original fully connected layer
        
        # Create new classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Save activations and gradients for Grad-CAM
        self.activations = None
        self.gradients = None
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Forward pass through the base model
        features = self.base_model(x)
        
        # Forward pass through the classifier
        return self.classifier(features)
    
    def activations_hook(self, grad):
        """Hook for saving gradients during backpropagation"""
        self.gradients = grad
    
    def get_activations(self, x):
        """
        Get activations from the last convolutional layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Activations tensor
        """
        # Forward pass until the last convolutional layer
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        # Save activations and register hook
        self.activations = x
        h = x.register_hook(self.activations_hook)
        
        return x
    
    def get_activations_gradient(self):
        """Get gradients of the activations"""
        return self.gradients
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        return predicted_classes, probabilities
