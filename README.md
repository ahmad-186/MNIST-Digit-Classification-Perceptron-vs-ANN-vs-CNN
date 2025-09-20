MNIST Digit Classification – Perceptron vs ANN vs CNN

Overview

I built and compared three approaches for MNIST handwritten digit classification:

  •	Perceptron (Single Layer)
  
  •	Artificial Neural Network (ANN / Multi-Layer Perceptron)
  
  •	Convolutional Neural Network (CNN)
  
The aim was to understand how performance improves as models become more sophisticated.

Repository Structure

│── Dataset

│── CNN_Project.ipynb     # Jupyter Notebook with full implementation

│── README.md             # Project documentation

Workflow

1.	Data Preprocessing
•	Loaded MNIST dataset (28×28 grayscale images).
•	Normalized pixel values [0,1].
•	Converted labels into categorical format.
2.	Visualization
•	Sample images plotted to better understand digit distribution.
3.	Model Implementations
•	Perceptron: Simple linear classifier.
•	ANN: Multi-layer perceptron with hidden layers & non-linear activations.
•	CNN: Used convolution, pooling, dropout, and dense layers for feature extraction.
4.	Evaluation Metrics
•	Accuracy
•	Confusion Matrix
•	Classification Report

Results

Model	Test Accuracy
Perceptron	~90.74%
ANN (MLP)	~97.69%
CNN	~99.21%

CNN achieved the best performance, proving the strength of convolutional layers for image classification.

Tech Stack

•	Python
•	NumPy, Pandas, Seaborn, Matplotlib
•	scikit-learn
•	TensorFlow / Keras

Run Locally

1.	Clone this repo:
2.	git clone https://github.com/yourusername/mnist-cnn-comparison.git
3.	cd mnist-cnn-comparison
4.	Install dependencies:
5.	pip install -r requirements.txt
6.	Open the Jupyter Notebook:
7.	jupyter notebook CNN_Project.ipynb

Learning Outcomes

•	Difference in performance between shallow (Perceptron) and deep models (ANN, CNN).
•	Why CNNs are the state-of-the-art for image classification.
•	Hands-on practice with Keras & TensorFlow.

Connect With Me

•	🌐 LinkedIn
•	💻 GitHub

This project is a milestone in my Deep Learning journey. More advanced Computer Vision projects coming soon! 🚀
