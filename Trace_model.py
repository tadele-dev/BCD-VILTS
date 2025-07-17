from app.model import MultimodalModel
import torch

model = MultimodalModel()
model.load_state_dict(torch.load("multimodal_model.pth"))  # if trained
model.eval()

example_image = torch.randn(1, 3, 224, 224)
example_input_ids = torch.randint(0, 1000, (1, 512))
example_attention_mask = torch.ones(1, 512)

traced = torch.jit.trace(model, (example_image, example_input_ids, example_attention_mask))
traced.save("traced_model.pt")
