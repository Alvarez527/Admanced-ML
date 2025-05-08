# 🧠 Deep Learning Foundations — From Fully Connected Nets to Transformers

Welcome to the **Deep Learning Foundations** repository! This curated collection of Jupyter notebooks guides you through key building blocks of modern artificial intelligence — starting from fully connected networks and progressing to transformers for natural language tasks.

Whether you're just starting out or aiming to deepen your understanding of practical implementations in deep learning, this repository provides a hands-on, code-first approach supported by theoretical insights.

---

## 📚 Learning Path & Notebook Guide

### 🔹 Module 1: Foundations of Fully Connected Networks

- `1a_Multilayer_Fully_Connected_Numpy.ipynb`  
  ➤ Build a fully connected neural network from scratch using NumPy. Learn the mathematics of forward and backward passes and grasp the intuition behind backpropagation.

- `1b_Fully_Connected_ASL_Dataset.ipynb`  
  ➤ Apply your fully connected model to the **American Sign Language (ASL)** image dataset. Explore preprocessing, data handling, and performance tuning.

---

### 🔹 Module 2: Deep Learning with PyTorch

- `2a_FC_ASL_using_pytorch.ipynb`  
  ➤ Transition from NumPy to **PyTorch**. Rebuild and train a fully connected model for the ASL dataset using high-level PyTorch APIs.

- `2b_CNN_Cifar_10_Using_Pytorch.ipynb`  
  ➤ Dive into **Convolutional Neural Networks (CNNs)** and apply them to the **CIFAR-10** image dataset. Understand how spatial hierarchies improve classification tasks.

- `2c_Transfer_Learning_CIFAR-10.ipynb`  
  ➤ Learn how to leverage pre-trained models through **transfer learning**, fine-tuning them on CIFAR-10 for faster convergence and better accuracy.

---

### 🔹 Module 3: Embeddings and Sequential Models

- `3a_Embeddings with Glove and Numpy.ipynb`  
  ➤ Use **GloVe** pre-trained word vectors to build meaningful word embeddings with NumPy. Understand the semantic relationships captured through vector space.

- `3b_Text_Classification_with RNNs and AG_News.ipynb`  
  ➤ Implement **Recurrent Neural Networks (RNNs)** for text classification on the **AG News** dataset. Learn about sequence modeling and temporal dependencies in language.

---

### 🔹 Module 4: Transformers and Advanced Architectures

- `4_Text_Translator_Using_Transformers.ipynb`  
  ➤ Explore **Transformer architectures** and build a simple text translator using attention mechanisms — the foundation of today’s state-of-the-art NLP systems.

- `Paper_Transformers Beyond NLP.pdf`  
  ➤ Bonus reading: A comprehensive paper on how **Transformers** extend beyond NLP, impacting vision, speech, and multimodal tasks.

---

## 🧪 Who Is This For?

This repository is ideal for:

- 👩‍💻 Beginners seeking an end-to-end learning path from scratch to state-of-the-art models.
- 🧑‍🏫 Educators or students exploring step-by-step implementations.
- 🛠️ Practitioners wanting to bridge theoretical foundations with practical PyTorch usage.

---

## 📁 Repository Structure

```bash
.
├── notebooks/
│   ├── 1a_Multilayer_Fully_Connected_Numpy.ipynb
│   ├── 1b_Fully_Connected_ASL_Dataset.ipynb
│   ├── 2a_FC_ASL_using_pytorch.ipynb
│   ├── 2b_CNN_Cifar_10_Using_Pytorch.ipynb
│   ├── 2c_Transfer_Learning_CIFAR-10.ipynb
│   ├── 3a_Embeddings with Glove and Numpy.ipynb
│   ├── 3b_Text_Classification_with RNNs and AG_News.ipynb
│   ├── 4_Text_Translator_Using_Transformers.ipynb
│   └── Paper_Transformers Beyond NLP.pdf
├── images/
│   └── (Visual aids and diagrams)
├── requirements.txt
└── README.md

```
## 🚀 Getting Started

1.-Clone this repository:
git clone https://github.com/Alvarez527/Advanced-ML.git
cd Advanced-ML

2.-(Optional) Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.-Install required packages:
pip install -r requirements.txt

4.-Launch Jupyter:
jupyter notebook

## 🎯 Learning Goals
By the end of this journey, you will be able to:

Grasp the architecture and math behind fully connected and convolutional networks.

Build and train models using NumPy and PyTorch.

Understand and apply word embeddings and RNNs to text data.

Use transfer learning and transformers for cutting-edge performance.

## 📎 Recommended Resources
Deep Learning Book by Goodfellow, Bengio & Courville

Stanford CS231n: CNNs for Visual Recognition

The Illustrated Transformer

PyTorch Documentation

## 🤝 Contributing
Have ideas, improvements, or corrections? Feel free to fork, submit issues, or open a pull request. Let's build this knowledge base together!





