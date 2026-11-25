
# Waste Classification using Machine Learning

A machine learning project that classifies waste images into six categories using traditional machine learning and image processing.

## Waste Categories
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Technologies Used
- Python
- OpenCV
- Scikit-learn
- Scikit-image
- NumPy
- Streamlit

## Feature Extraction
Three image-processing techniques are used:

1. HOG (Histogram of Oriented Gradients)
2. LBP (Local Binary Pattern)
3. HSV Color Histogram

The extracted features are classified using a Random Forest model.

## Web Application
The Streamlit application allows users to upload a waste image and view its predicted category.

## Setup Instructions

1. Clone the repository:
   `git clone https://github.com/msdalal13/waste_classification_ml.git`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Place the trained model file `waste_random_forest.pkl` in the project directory.

4. Run the application:
   `streamlit run app.py`

## Authors
- Mihir Dalal
- Ankush Patil
