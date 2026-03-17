# project-Follicular-Lymphoma-Classification
**Follow these steps to set up and launch the Jupyter Notebook environment for this project:**
1. Activate Virtual Environment
venv\Scripts\activate
This command activates the virtual environment (venv) for the project.
It ensures all dependencies (libraries, packages) are isolated from your global Python installation.
After activation, your terminal will show (venv) at the beginning.

2. Navigate to the Notebook Directory
cd .\Notebook\
This moves into the Notebook folder where the Jupyter notebooks are stored.

3. Launch Jupyter Notebook
jupyter notebook
This starts the Jupyter Notebook server.
It will automatically open in your default browser.
You can now open and run .ipynb files interactively.

**Following are the libraries used in code**
numpy – Fundamental library for numerical computing with arrays and mathematical operations.
pandas – Data analysis and manipulation library for structured datasets (DataFrames).
flask – Lightweight web framework for building APIs and web applications.
Pillow – Image processing library for opening, editing, and saving images.
tqdm – Displays progress bars for loops and long-running tasks.
opencv-python – Computer vision library for image and video processing.
scikit-learn – Machine learning library with tools for classification, regression, and clustering.
scikit-image – Image processing library built on NumPy for advanced image analysis.
scipy – Scientific computing library for optimization, integration, and statistics.
pickle-mixin – Utility for serializing and saving Python objects.
notebook – Jupyter Notebook interface for interactive coding and visualization.
matplotlib – Plotting library for creating static, animated, and interactive graphs.
seaborn – Statistical data visualization library built on Matplotlib.
plotly – Interactive visualization library for web-based graphs and dashboards.
torch – Deep learning framework (PyTorch) for building neural networks and AI models.
tensorflow – End-to-end machine learning framework for training and deploying models.
keras – High-level API for building and training deep learning models (runs on TensorFlow).
h5py – Interface for storing and managing large datasets in HDF5 file format.

Cell 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
numpy → used for numerical operations (arrays, matrices).
pandas → for handling structured data (like CSV files).
matplotlib.pyplot → used for plotting graphs and images.
cv2 (OpenCV) → for image processing (reading, resizing, etc.).
os → helps interact with file system (folders, file paths).

🔹 Cell 2: Skimage & GLCM Imports
from skimage.feature import graycomatrix, graycoprops
graycomatrix → creates GLCM (Gray-Level Co-occurrence Matrix).
graycoprops → extracts texture features like:
Contrast
Energy
Homogeneity
Correlation
This is the texture analysis part of your project.

🔹 Cell 3: TensorFlow / Keras Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
Sequential → linear stack of layers.
Conv2D → convolution layer (extracts image features).
MaxPooling2D → reduces image size (downsampling).
Flatten → converts 2D feature maps → 1D vector.
Dense → fully connected layer.
Dropout → prevents overfitting.

🔹 Cell 4: Dataset Path
data_dir = "path_to_dataset"
Defines location of dataset.
Usually structured like:
dataset/
   class1/
   class2/
🔹 Cell 5: Image Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray
Line-by-line:
cv2.imread() → loads image.
cv2.resize() → resizes to fixed size (128×128).
cv2.cvtColor() → converts image to grayscale.
Returns:
RGB image (for CNN)
grayscale image (for GLCM)

🔹 Cell 6: GLCM Feature Extraction
def extract_glcm_features(gray):
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
Creates GLCM matrix:
[1] → pixel distance
[0] → angle (0° direction)
contrast = graycoprops(glcm, 'contrast')[0, 0]
Measures intensity variation.
energy = graycoprops(glcm, 'energy')[0, 0]
Measures uniformity.
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
Measures closeness of pixel distribution.
correlation = graycoprops(glcm, 'correlation')[0, 0]
Measures how correlated pixels are.
return [contrast, energy, homogeneity, correlation]
Returns all features as a list.

🔹 Cell 7: Data Loading Loop
images = []
features = []
labels = []
Initialize lists to store:
Images
Texture features
Labels
for folder in os.listdir(data_dir):
Loop through each class folder.
for file in os.listdir(os.path.join(data_dir, folder)):
Loop through images inside folder.
img, gray = preprocess_image(path)
Preprocess image.
glcm_feat = extract_glcm_features(gray)
Extract texture features.
images.append(img)
features.append(glcm_feat)
labels.append(folder)
Store data.

🔹 Cell 8: Convert to Arrays
images = np.array(images)
features = np.array(features)
labels = np.array(labels)
Convert lists → NumPy arrays for ML model.

🔹 Cell 9: Normalize Images
images = images / 255.0
Scales pixel values:
From [0–255] → [0–1]
Helps faster training.

🔹 Cell 10: Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

Converts class names → numeric labels.

🔹 Cell 11: Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
Splits dataset:
80% training
20% testing

🔹 Cell 12: CNN Model Creation
model = Sequential()
Initializes model.
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
First convolution layer:
32 filters
3×3 kernel
model.add(MaxPooling2D(pool_size=(2,2)))
Reduces image size.
model.add(Conv2D(64, (3,3), activation='relu'))
Second convolution layer.
model.add(MaxPooling2D(pool_size=(2,2)))
Further downsampling.
model.add(Flatten())
Converts feature maps → vector.
model.add(Dense(128, activation='relu'))
Fully connected layer.
model.add(Dropout(0.5))
Prevents overfitting.
model.add(Dense(1, activation='sigmoid'))
Output layer (binary classification).

🔹 Cell 13: Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Adam → adaptive optimizer.
binary_crossentropy → loss function.
Tracks accuracy.

🔹 Cell 14: Train Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
Trains model:
10 epochs
Validates on test set.

🔹 Cell 15: Evaluate Model
loss, acc = model.evaluate(X_test, y_test)
Computes final:
Loss
Accuracy

🔹 Cell 16: Plot Results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
Plots training vs validation accuracy.
plt.legend(['Train', 'Validation'])
plt.show()


