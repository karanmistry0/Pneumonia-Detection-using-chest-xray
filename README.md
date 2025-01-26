# Pneumonia Detection using Chest X-Ray

This project aims to detect pneumonia from chest X-ray images using a convolutional neural network (CNN). The dataset used is sourced from [Kaggle's Chest X-Ray Images dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [images](#images)
- [Future Enhancements](#future-enhancements)
  

## Project Overview
This project leverages deep learning techniques to classify chest X-ray images into two categories:
- Normal
- Pneumonia

A Flask web application is used to deploy the trained model, enabling users to upload chest X-ray images and get predictions.

## Technologies Used
- Python
- TensorFlow/Keras for CNN model development
- Flask for web application deployment
- HTML and CSS for web page development
- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Model Performance
The CNN model achieved the following performance metrics:
- **Training Accuracy:** 98.35%
- **Validation Accuracy:** 87.50%
- **Test Accuracy:** 91.40%

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone "https://github.com/shubham15400/Machine_Learning_Framework/"
   cd Machine_Learning_Framework
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
4. Train the model (if required):
   ```bash
   python Project.ipynb
   ```
   The pre-trained model is included in the repository for convenience.
5. Run the Flask application:
   ```bash
   python app.py
   ```
6. Open the web application in your browser at `http://127.0.0.1:5000`.

## Usage
1. Launch the Flask web application.
2. Upload a chest X-ray image using the interface.
3. View the prediction result (Normal or Pneumonia).

## Images

1. Normal healthy lung X-ray. <br> <img src="Sample images/Normal.jpeg" width="350" title="Normal healthy lung X-ray"/> <br> <br>
2. Pneumonia X-ray. <br> <img src="Sample images/Pneumonia.jpeg" width="350" title="Pneumonia X-ray"/>

## Future Enhancements
- Improve model accuracy by experimenting with advanced architectures.
- Add support for multi-class classification (e.g., distinguishing between bacterial and viral pneumonia).
- Enable deployment on cloud platforms such as AWS, Google Cloud, or Azure.
- Enhance the user interface for a better user experience.

---

Feel free to contribute to this project by submitting issues or pull requests.
