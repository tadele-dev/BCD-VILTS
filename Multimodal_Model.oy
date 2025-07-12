import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import AutoModel, AutoTokenizer


class MultimodalModel(nn.Module):
    def __init__(self, num_classes=2, text_model_name="bert-base-uncased", hidden_dim=512):
        """
        Initialize the multimodal model with CNN for image processing and a Transformer-based
        text encoder for clinical notes.
        
        Args:
            num_classes (int): Number of output classes.
            text_model_name (str): Hugging Face transformer model name for text encoding.
            hidden_dim (int): Hidden dimension size for the fused representation.
        """
        super(MultimodalModel, self).__init__()

        # Image encoder: CNN (ResNet-50 backbone)
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()  # Remove classification head to get raw features

        # Text encoder: Transformer (e.g., BERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        # Multimodal fusion with Transformer
        self.fusion_transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, texts):
        """
        Forward pass for multimodal input data.
        
        Args:
            images (torch.Tensor): Batch of image tensors (B, C, H, W).
            texts (list): List of strings (clinical notes).
            
        Returns:
            torch.Tensor: Predicted class logits.
        """
        # Extract image features using CNN
        image_features = self.image_encoder(images)  # (B, 2048)
        image_features = image_features.unsqueeze(1)  # Add sequence dimension: (B, 1, 2048)

        # Tokenize and encode text using the Transformer
        text_inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_inputs = {k: v.to(images.device) for k, v in text_inputs.items()}  # Move text inputs to GPU/CPU
        text_outputs = self.text_encoder(**text_inputs)
        text_features = self.text_fc(text_outputs.last_hidden_state[:, 0, :])  # CLS token: (B, hidden_dim)

        # Combine image and text features into a single sequence
        multimodal_input = torch.cat([image_features, text_features.unsqueeze(1)], dim=1)  # (B, 2, hidden_dim)

        # Pass the fused input through the Transformer
        fused_output = self.fusion_transformer(multimodal_input, multimodal_input)  # (B, 2, hidden_dim)

        # Classification based on the output of the first token (image features)
        logits = self.classifier(fused_output[:, 0, :])  # (B, num_classes)

        return logits


if __name__ == "__main__":
    # Initialize the model
    model = MultimodalModel(num_classes=2)
    model.eval()  # Switch to evaluation mode

    # Sample input data
    sample_images = torch.randn(2, 3, 224, 224)  # Batch of 2 images (B, C, H, W)
    sample_texts = [
        "The patient has a high probability of malignancy based on prior biopsy results.",
        "Benign tumor detected with no signs of aggressive growth."
    ]

    # Perform inference
    with torch.no_grad():
        output = model(sample_images, sample_texts)
        print("Predicted logits:", output)
