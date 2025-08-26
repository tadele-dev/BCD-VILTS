
# 🧠 BCD-VILTS: Breast Cancer Detection using Vision-and-Language Transformer System

A next-generation multimodal AI model for early and accurate breast cancer detection by integrating **medical imaging** and **clinical text reports** using a hybrid **CNN + Transformer** architecture.

📘 Project Definition

BCD-VILTS (Breast Cancer Detection using Vision-and-Language Transformer System) is a novel, independently developed deep learning model that integrates medical imaging and clinical text data for improved breast cancer detection. While the architecture draws conceptual inspiration from the ViLT (Vision-and-Language Transformer) model—particularly in its approach to multimodal fusion—it is a standalone implementation designed specifically for the healthcare domain.

> This project, BCD-VILTS, is an independent implementation inspired by the ViLT architecture. It is not affiliated with, endorsed by, or derived from the official ViLT authors. Any similarity in naming is purely to acknowledge the inspiration in multimodal fusion techniques, and the model has been developed with original code and domain-specific objectives.

# Multimodal Transformer with Image Anchor

A novel multimodal architecture designed for clinical decision support that fuses visual and textual data using a simplified Transformer model. The image token serves as the anchor for classification after being contextually enriched by the clinical text. This model is efficient, modular, and ideal for real-world deployment.

---

## 🚀 Statement of Novelty: Multimodal Transformer Model

### 1. Simplified Yet Effective Fusion Mechanism
Unlike many existing models that use complex cross-attention, co-attention, or contrastive learning strategies to fuse image and text, this model uses a **single encoder-decoder `nn.Transformer` block** with only two input tokens:

- One token from a **ResNet-encoded image**.
- One token from a **BERT-encoded clinical note**.

> 🔑 This avoids the architectural complexity and memory overhead found in multi-headed or region-token systems like ViLBERT or LXMERT.

---

### 2. Image Token as a Classification Anchor
The model classifies using the **image token** after it has been contextually enriched via self-attention with the text token. This approach contrasts with:

- Models that concatenate fused features for classification.
- Models that treat text as the main classification token.

> 🔑 By selecting the image token as the decision anchor, the model maintains a **visual focus** while still being **text-aware** — critical for clinical tasks like radiology or pathology interpretation.

---

### 3. No Fine-Grained Alignment Required
Existing methods often require:
- ROI (Region of Interest) bounding boxes for image-text alignment.
- Word-region alignment annotations.
- Image patch tokenization (e.g., ViT + BERT).

This model operates **without such requirements**. It fuses **global image and text features**, making it lighter, data-efficient, and suitable for settings with limited annotations.

> 🔑 The design supports low-resource or emerging clinical environments without relying on high-cost dataset preprocessing.

---

### 4. Modular, Plug-and-Play Architecture
The model separates:
- Visual backbone (ResNet)
- Text encoder (BERT)
- Fusion layer (Transformer)
- Classifier head

Each can be swapped independently. This modularity makes the architecture **highly reusable and generalizable** to other medical tasks (e.g., X-ray + report, histopathology + biopsy).

---

### 5. Readiness for On-Device Deployment
With only **two tokens** processed through the transformer, the memory and compute footprint is minimal. The architecture is therefore compatible with:

- Edge devices in hospitals.
- Mobile diagnostic tools.
- Embedded AI applications in medical imaging hardware.

---

## 🧱 Core Components

- **ResNet-50**: Pretrained encoder for image features (2048-D).
- **BERT**: Pretrained encoder for clinical text (CLS token).
- **Transformer**: Self-attention encoder with 6 layers.
- **Classifier**: Feedforward head using image token.

---

## 🧠 Applications

- Breast cancer risk classification from mammograms + notes
- Radiology triage using imaging and report
- Pathology fusion from slide + biopsy summary

---

## 📜 License
This project is licensed under the
Apache License 2.0

2025 Tadele Tatek Gebrewold.
---

## ✍️ Author
**Tadele Tatek Gebrewold**  
📧 tadele_tatek@yahoo.com  
🌍 Addis Ababa, Ethiopia
