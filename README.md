## 📚 Project Overview

Food classification is an important application in AI-powered nutrition tracking and dietary analysis. This project trains a ResNet-50 model to classify 11 food categories with high accuracy. I fine-tuned the model and used Grad-CAM to visualize predictions.

📌 Key Features:  

🔹 Pre-trained ResNet-50 with fine-tuning  
🔹 Food-11 Dataset for classification  
🔹 Grad-CAM visualization for interpretability  
🔹 Streamlit app for real-time predictions  

## 🗂️ Dataset Information

The Food-11 dataset consists of a training set with a total of 9,900 images, each class containing 900 images, and a test set with a total of 1,100 images, each class containing 100 images, making a total of 11,000 images.

📌 Classes:    

🥧 **Apple Pie**, 🍰 **Cheese Cake**, 🍛 **Chicken Curry**, 🍟 **French Fries**, 🍚 **Fried Rice**,  
🍔 **Hamburger**, 🌭 **Hot Dog**, 🍦 **Ice-cream**, 🍳 **Omelette**, 🍕 **Pizza**, 🍣 **Sushi**


🔁 Download Dataset: [Food-11 Dataset](https://www.kaggle.com/datasets/imbikramsaha/food11) (Put the dataset in data directory)


## 🛠️ Model Architecture

The model is based on ResNet-50 with a custom classifier:

```
(fc): Sequential(
  (0): Linear(in_features=2048, out_features=512, bias=True)
  (1): BatchNorm1d(512)
  (2): ReLU()
  (3): Dropout(p=0.3)
  (4): Linear(in_features=512, out_features=11, bias=True)
)
```

📌 Why ResNet-50?  
  
🔹 Transfer learning for better performance  
🔹 Handles complex food textures  
🔹 Efficient feature extraction  

## ⚙️ Installation

1. Clone the repository:

```
>>> git clone https://github.com/yourusername/Food-11-Classification.git
>>> cd Food-11-Classification
```

2. Create Virtual Environment and install requried libraries

```
>>> python -m venv myenv
>>> myenv\Scripts\activate.bat
>>> pip install -r requirements.txt
```

## 🚀 Training & Inference

**For Training:**

```
python train.py --train_folder ./data/train/ --epochs 10 --batch_size 32 --lr 0.0001
```

**For Inference:**

```
python inference.py --model_path ./models/finetuned_resnet50.pth --input_folder ./inputs/ --output_folder ./results/
```

Download Finetuned model: [Finetuned_Resnet50.pth](https://drive.google.com/uc?export=download&id=1J5rgk2rBY7a8WGjuvTnd3p1C55pB-q87) (Put this model in models directory)

## 📊 Results  

| Model                  | Accuracy  |
|:----------------------:|:---------:|
| **Baseline (ResNet-50)**  | **14.73%**   |
| **Training** | **99.53%** |
| **Validation** | **82.02%** |
| **Testing** | **81.45%** |


📌 **Note:** To check the baseline accuracy, run the following script:  
```
python baseline_evaluation.py
```

## 🎬 Demo  

### 🔹 Running FastAPI Backend  
Start the FastAPI server using the following command:  

```
python src/fastapi/app.py
```

### 🔹 Running Streamlit UI 
Start the streamlit using the following command:  

```
python src/streamlit/main.py
```
