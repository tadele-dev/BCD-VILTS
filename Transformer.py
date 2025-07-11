import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import AutoModel, AutoTokenizer

class MultimodalTransformer(nn.Module):
    def __init__(self, num_classes=2, text_model_name="bert-base-uncased", hidden_dim=512):
        super(MultimodalTransformer, self).__init__()

        # Image feature extractor (ResNet backbone)
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()  # Remove the classification head

        # Text feature extractor (BERT or similar)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        # Transformer encoder for multimodal fusion
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
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

    def forward(self, image, text):
        """
        Args:
            image (torch.Tensor): Batch of image tensors (B, C, H, W).
            text (list): List of strings containing clinical notes.
        Returns:
            torch.Tensor: Predicted class logits.
        """

        # Image embedding
        image_features = self.image_encoder(image)  # (B, 2048)
        image_features = image_features.unsqueeze(1)  # Add sequence dimension -> (B, 1, 2048)

        # Text embedding
        text_inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_outputs = self.text_encoder(**{k: v.to(image.device) for k, v in text_inputs.items()})
        text_features = self.text_fc(text_outputs.last_hidden_state[:, 0, :])  # CLS token -> (B, hidden_dim)

        # Combine modalities into a sequence
        multimodal_input = torch.cat([image_features, text_features.unsqueeze(1)], dim=1)  # (B, 2, hidden_dim)

        # Pass through the transformer
        multimodal_output = self.transformer(multimodal_input, multimodal_input)  # (B, 2, hidden_dim)

        # Classification using the output of the first token (image token)
        logits = self.classifier(multimodal_output[:, 0, :])  # (B, num_classes)

        return logits

# Example usage
if __name__ == "__main__":
    # Model initialization
    model = MultimodalTransformer(num_classes=2)
    model.eval()

    # Sample inputs
    sample_image = torch.randn(2, 3, 224, 224)  # Batch of 2 RGB images
    sample_text = ["Patient history suggests a high risk of breast cancer.", 
                   "Clinical notes indicate benign tumor characteristics."]

    # Forward pass
    with torch.no_grad():
        output = model(sample_image, sample_text)
        print("Logits:", output)
