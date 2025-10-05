# 🧠 Brain Tumor MRI Classifier

A deep learning web application built using **TensorFlow** and **Streamlit** to classify brain tumor MRI scans into four categories: Glioma, Meningioma, Pituitary, and No Tumor.

---

## 🚀 Live Demo
Access the app here:  
[Streamlit App Link](https://yourusername-yourappname.streamlit.app)

---

## 🧩 Project Overview
This project demonstrates an end-to-end workflow for **medical image classification**:
1. Preprocessing MRI scans
2. Training a **TensorFlow EfficientNetV2B0** model
3. Building a **Streamlit web interface** for real-time predictions

The app allows users to upload MRI images and get a **predicted tumor type** along with a **confidence score**.

---

## 📊 Dataset
- Dataset: Brain tumor MRI dataset with 4 classes
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- Each class contains MRI images stored in separate folders.
- Example images included in `data/` folder.

---

## 🛠 Tech Stack
- **TensorFlow / Keras** – Model training and inference  
- **Streamlit** – Web application interface  
- **OpenCV & Pillow** – Image preprocessing  
- **Google Colab** – Free GPU for model training  
