# Maya 🤖

> **An educational and engineering project that implements a Transformer-based conversational AI from first principles using Python and NumPy.**

Maya is an open-source project focused on understanding and implementing modern Natural Language Processing architectures from scratch. Instead of relying on deep learning frameworks, Maya reimplements the core components of a Transformer-based conversational AI to provide a transparent and educational codebase for learning, experimentation, and future research.

The long-term vision is to evolve Maya into a modular AI assistant capable of intelligent routing, retrieval, memory, and tool integration while keeping the core architecture implemented from first principles.

---

# 🚀 Key Highlights

- 🧠 **2.91 Million** trainable parameters
- ⚙️ **3 Encoder + 3 Decoder** Transformer architecture
- 🔤 Custom **Byte Pair Encoding (BPE)** tokenizer
- 🎯 **Greedy Search** & **Beam Search** decoding
- 📚 Trained on **~16,000** conversational samples
- 🐍 Implemented entirely using **Python + NumPy**
- 🌐 Flask-based web interface

---

# 📊 Model Specifications

| Component | Specification |
|-----------|---------------|
| **Architecture** | Encoder–Decoder Transformer |
| **Trainable Parameters** | **2,914,184 (~2.91 Million)** |
| **Encoder Layers** | 3 |
| **Decoder Layers** | 3 |
| **Embedding Dimension** | 128 |
| **Feed Forward Dimension** | 256 |
| **Attention Heads** | 4 |
| **Vocabulary Size** | ~5,000 |
| **Tokenizer** | Custom Byte Pair Encoding (BPE) |
| **Decoding Strategy** | Greedy Search & Beam Search |
| **Framework** | Pure Python + NumPy |

---

# 📚 Training Dataset

Maya V1 was trained on approximately **16,000 conversational samples** collected from multiple conversational datasets.

| Dataset | Samples |
|----------|---------:|
| Greetings & Daily Conversations | ~4,000 |
| DailyDialog | ~7,000 |
| Reddit Conversations | ~5,000 |

The combined dataset focuses on open-domain dialogue, enabling Maya to respond naturally to greetings, casual discussions, and everyday conversations.

---

# 🛠️ Implemented From Scratch

## Transformer

- Encoder
- Decoder
- Multi-Head Attention
- Positional Encoding
- Residual Connections
- Layer Normalization
- Feed Forward Networks

## Tokenization

- Byte Pair Encoding (BPE)
- Vocabulary Generation
- Encoding & Decoding Pipeline

## Training

- Cross Entropy Loss
- Adam Optimizer
- Backpropagation
- Model Serialization

## Text Generation

- Greedy Search
- Beam Search

## Deployment

- Flask Backend
- Browser-based Chat Interface

## 📸 Screenshots

### Greeting Conversation

![Greeting](Maya_v1/images/chatbot1.png)

### Sample Conversation

![Conversation](Maya_v1/images/chatbot2.png)

---

# ⚠️ Current Limitations

Maya V1 is an early-stage implementation whose primary goal is to learn and implement Transformer architectures from first principles rather than to compete with production-scale language models.

Current limitations include:

- Limited training corpus (~16K conversational samples)
- Responses may occasionally be repetitive or lack contextual understanding
- Limited factual and domain-specific knowledge
- No retrieval or external knowledge integration
- No long-term conversational memory
- Reasoning capabilities are limited compared to modern LLMs

Despite these limitations, Maya V1 demonstrates a complete end-to-end implementation of a Transformer-based conversational AI, including custom tokenization, training, inference, and decoding implemented entirely in Python and NumPy.

----

# 📂 Repository Structure

```
Maya/
│
├── README.md
├── Demo.md
│
├── Maya_v1/
│   ├── app.py
│   ├── chatbot_engine.py
│   ├── requirements.txt
│   ├── templates/
│   ├── static/
│   ├── maya_v1.pkl
│   └── ...
│
├── Maya_v2/
│
└── Maya_v3/
```

---

# ⚡ Installation

Clone the repository.

```bash
git clone https://github.com/alokkumar-project/Maya.git
```

Navigate to Maya V1.

```bash
cd Maya/Maya_v1
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the application.

```bash
python app.py
```

Open your browser.

```
http://127.0.0.1:5000
```

---

# 🎥 Demo

A demonstration of Maya V1 is available in **Demo.md**.

---

# 🗺️ Roadmap

## ✅ Maya V1

Current release featuring:

- Transformer-based conversational chatbot
- Custom BPE tokenizer
- Beam Search & Greedy Search
- Flask web application
- End-to-end training pipeline

---

## 🚧 Maya V2

Planned improvements:

- Improved conversational quality
- Larger and more diverse conversational datasets
- Better decoding strategies
- Faster inference
- Improved Transformer architecture
- Enhanced training pipeline

---

## 🚧 Maya V3

Long-term objectives:

- Intent Classification
- Expert Model Routing
- Retrieval-Augmented Generation (RAG)
- Long-Term Memory
- Database Integration
- External API Integration
- Tool Calling
- Modular AI Assistant Architecture

---

# 🎯 Project Goals

Maya aims to provide a clean and educational implementation of modern NLP architectures while gradually evolving into a modular AI assistant.

Future work focuses on:

- Improving dialogue generation
- Expanding model capabilities
- Integrating retrieval and reasoning modules
- Supporting multiple expert models
- Maintaining an easy-to-understand codebase for learning and experimentation

---

# 🤝 Contributing

Contributions, suggestions, and discussions are welcome.

If you discover a bug or have ideas for improving Maya, feel free to open an issue or submit a pull request.

---

# 👨‍💻 Author

**Alok Kumar**

Student, IIT (ISM) Dhanbad

GitHub: https://github.com/alokkumar-project
