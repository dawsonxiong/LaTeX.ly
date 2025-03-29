# Welcome to LaTeX.ly!

<img src="https://github.com/dawsonxiong/LaTeX.ly/blob/main/frontend/public/home.png" alt="Homepage" width="1120" height="430">

## 🚀 Features
- Math equation uploading and processing
- Accurate and effective math symbol recognition (OCR)
- LaTeX math mode styling display
- Conversion history
- RESTful API architecture
- Clean and User-Friendly React UI

## 🛠 Built with
### Backend
- Python:
  - PyTorch
  - OpenCV
  - Matplotlib
  - Flask
  - Numpy
- Jupyter Notebook
- Docker (implementing)

### Frontend
- JavaScript:
  - React
  - Next.js
- Tailwind CSS

## 🔧 Installation & Usage
### Prerequisites:
- Node.js (v14.6.0 or higher)
- Docker
- Python (v3.9 or higher)

## ❗ What I learned
### Having a good dataset is important
- Ultimately, I ended up
  - Creating **1.6k original images** of 64 math symbols in 25 different fonts using Matplotlib
  - Applied minor transformations, creating **24k+ augmented images**
  - Sourcing **25k handwritten math symbols** from CROHME
- Ultimately, I ended up with **49k+ unique images of math symbols**
- Throughout the process, I had to consistently find and create more data, without overfitting

![Dataset](https://github.com/dawsonxiong/LaTeX.ly/blob/main/frontend/public/dataset.png)
Almost 50k images!

### Preprocessing data is key
- Data, especially handwritten math symbols, contain many inconsistencies
- With a convolutional neural network trained on consistent data, it is necessary to process data before usage
- For instance, my algorithm applies a Gaussian blur, adaptive thresholding, and a denoising filter amongst other things
- Without preprocessing data, the model's accuracy significantly decreases

<img src="https://github.com/dawsonxiong/LaTeX.ly/blob/main/frontend/public/prediction.png" alt="Prediction" width="220" height="210">

### How Neural Networks work
- With this project, I got to work with and understand how CNNs take in data and output results
- I learned:
  - How to structure a neural network for OCR tasks, balancing the number of layers and neurons for training
  - The importance of choosing the right activation functions (i.e. ReLU improved symbol detection, while softmax was useful for classification)
  - How to use SGD with mini-batches for efficiency and Cross Entropy Loss to improve accuracy

### It's okay to start over
- I originally started this project in November, deciding to use Tesseract, a pre-built open source OCR model
- However, the model was trained only on English letters, making math symbol recognition impossible
- After trying to train my own Tesseract model, the mix of its dataset and mine was not effective either
- I then experimented with Tensorflow and Pytorch, ultimately choosing the latter for its simplicity
- Having restarted my project, I spent many hours cultivating a dataset, which I then used to train my model
- But in my first four attempts, my model had surprisingly low accuracy
  - In the CROHME dataset, there were many innacurate symbols
  - After slowly cleaning my dataset, 5 models later, I had created one that was **90%+ more accurate than Tesseract**
- Approaching and learning something new is scary. But I learned not to be afraid to restart, if it means getting on the right track
