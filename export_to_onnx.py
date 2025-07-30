import torch
from bcdvilts_model import BCDVILTSModel

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "bcdvilts_trained.pth"
ONNX_PATH = "bcdvilts.onnx"
BATCH_SIZE = 1
IMAGE_DIM = (3, 224, 224)
TEXT_SEQ_LEN = 512

# -------------------------------
# Load model
# -------------------------------
model = BCDVILTSModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# -------------------------------
# we Create dummy inputs
# -------------------------------
dummy_image = torch.randn(BATCH_SIZE, *IMAGE_DIM)
dummy_text = ["Patient shows signs of dense breast tissue."]

# Important: ONNX export only accepts **Tensors** as inputs
# So we override the model forward to accept raw tensors (image + tokenized text)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=TEXT_SEQ_LEN)

# -------------------------------
# Modify forward temporarily for ONNX
# -------------------------------
class ExportableModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, image, input_ids, attention_mask, token_type_ids):
        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return self.model(image, text_inputs)

export_model = ExportableModel(model, tokenizer)

# -------------------------------
# Export
# -------------------------------
torch.onnx.export(
    export_model,
    (dummy_image, text_inputs["input_ids"], text_inputs["attention_mask"], text_inputs["token_type_ids"]),
    ONNX_PATH,
    input_names=["image", "input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "token_type_ids": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=12
)

print(f"Exported ONNX model to: {ONNX_PATH}")
