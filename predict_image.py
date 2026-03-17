import os
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler

def preprocess_image(image_path, target_size):
    """
    Load and preprocess the image for CNN model feature extraction.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

##=============================##

def extract_glcm_features(image, cnn_model, output_csv, distances, angles):
        
    # CNN Features
    image_expanded = np.expand_dims(image, axis=0)  # Expand dimensions for the CNN model
    cnn_features = cnn_model.predict(image_expanded).flatten()

    # GLCM Features
    image_gray = np.mean(image, axis=-1).astype(np.uint8)  # Convert to grayscale if needed
    glcm_features = calculate_glcm_features(image_gray, distances, angles)

    # Combine CNN + GLCM features
    return np.concatenate([cnn_features, glcm_features])

##=============================##

def predict_from_image(image_path, cnn_model, ann_model, scaler, distances, angles, target_size):
    """
    Predict the class of an image using CNN and ANN models with GLCM features.
    """
    # Step 1: Extract CNN features
    cnn_input = preprocess_image(image_path, target_size)
    cnn_features = cnn_model.predict(cnn_input).flatten()

    # Step 2: Extract GLCM features
    img = load_img(image_path, color_mode="grayscale", target_size=target_size)
    img_array = img_to_array(img).astype(np.uint8).squeeze()
    glcm_features = extract_glcm_features(img_array, distances, angles)

    # Step 3: Combine CNN and GLCM features
    combined_features = np.concatenate([cnn_features, glcm_features]).reshape(1, -1)

    # Step 4: Standardize the features
    standardized_features = scaler.transform(combined_features)

    # Step 5: Predict using the ANN model
    predictions = ann_model.predict(standardized_features)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return predicted_class

if __name__ == "__main__":
    # Paths
    image_path = os.path.join('..', 'Dataset', 'data', 'unseen', '2', '044996CR__20241203_121010.tiff') # Image for prediction
    saved_image_path = os.path.join('..', 'Dataset', 'data', 'unseen', '2', 'Saved_044996CR__20241203_121010.jpg') 
    
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    # Open and save as JPEG or PNG
    img = Image.open(image_path)
    resized_img = img.resize((224, 224))  # Adjust dimensions as needed
    resized_img.save(saved_image_path)
    # img.save(saved_image_path, "JPEG")
    
    image_path = saved_image_path
    
    # Load models
    cnn_model = load_model(CNN_MODEL)
    ann_model = load_model(ANN_MODEL)
    scaler = load_model(StandardScaler_MODEL)  # Assuming you saved StandardScaler using joblib

    # GLCM parameters
    distances = [1, 3, 5, 3, 1, 3, 5]
    angles = [0, 0, 0, np.pi/4, np.pi/2, np.pi/2, np.pi/2]

    # Predict
    target_size = (128, 128)  # Size used in training
    prediction = predict_from_image(image_path, cnn_model, ann_model, scaler, distances, angles, target_size)

    print(f"The predicted class for the image is: {prediction}")