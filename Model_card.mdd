# BCD-VILTS: Breast Cancer Detection via Vision-and-Language Transformer with Self-Attention Fusion

## 🧠 Model Overview
BCD-VILTS is a custom multimodal AI model for breast cancer detection that fuses visual (image) and textual (clinical notes) data using a Transformer-based fusion block. It is designed to enhance diagnostic accuracy by modeling complex interactions between radiological imagery and corresponding clinical reports.

---

## 💡 Key Features

- **Multimodal Fusion**: Combines image and text using deep self-attention fusion.
- **Transformer-Based Architecture**: Utilizes ResNet-50 as an image encoder, BERT as a text encoder, and a multi-layer Transformer for fusion.
- **Image as [CLS] Token**: Treats visual embeddings as the primary anchor for classification.
- **Modality Drop Tolerance**: Robust under missing or incomplete modality inputs.
- **Explainability**: Provides cross-modal attention maps for interpretability.
- **Edge-Ready**: ONNX-compatible for mobile and edge deployment.

---

## 🔍 Applications

- Early and accurate breast cancer detection
- Decision support in radiology
- Explainable AI for clinical workflows
- Research in multimodal medical AI

---

## 🧪 Model Architecture

- `Image Encoder`: ResNet-50 pretrained on ImageNet
- `Text Encoder`: BERT-base pretrained on PubMed abstracts
- `Fusion Module`: Multi-head cross-attention Transformer (PyTorch `nn.Transformer`)
- `Classifier`: Fully-connected layer on top of fused image token

---

## 📊 Benchmark Results

| Dataset        | Accuracy | F1-Score | AUC    |
|----------------|----------|----------|--------|
| Custom Dataset | 0.91     | 0.89     | 0.94   |
| (Add details)  | (TBD)    | (TBD)    | (TBD)  |

---

## 🔬 Comparison with Related Models

| Model       | Modality        | Focus                           | BCD-VILTS Advantage                  |
|-------------|------------------|----------------------------------|--------------------------------------|
| **BioGPT**  | Text only        | Biomedical text generation and QA | BCD-VILTS handles both image and text |
| **MedCLIP** | Image + Text     | Radiology report-image alignment | BCD-VILTS performs classification with deeper fusion |
| **GatorTron**| Clinical text    | EHR understanding and cohorting  | BCD-VILTS focuses on image-text fusion for diagnosis |
| **ViLT**     | Image + Text    | General V+L understanding         | BCD-VILTS is tailored for breast cancer with custom architecture |
| **CLIP**     | Image + Text    | Contrastive learning, retrieval  | BCD-VILTS uses attention-based classification instead of contrastive loss |

---

## 🗂️ Input and Output Formats

**Input**:
- Image: 3-channel breast scan (224×224)
- Text: Clinical note (tokenized using BERT tokenizer)

**Output**:
- Binary label: `0 = No cancer`, `1 = Cancer detected`
- Optional: Attention heatmaps

---

## 📦 Export and Deployment

- Exportable to **ONNX** for mobile/web/edge
- Compatible with **TorchScript**, **TensorRT**, and **OpenVINO**

---

## 📈 Future Work

- Add pathology/genomic data for tri-modal fusion
- Evaluate on public datasets like CBIS-DDSM or INbreast
- Submit to **MICCAI**, **NeurIPS**, or **Nature Scientific Reports**

---

## 📚 Citation
