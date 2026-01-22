import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2 as cv

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

model = load_model('model.classifier.keras')

image = cv.imread('horse.png')
if image is None:
    raise FileNotFoundError("Image file not found")

image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_resized = cv.resize(image_rgb, (32,32))

prediction = model.predict(np.array([image_resized]) / 255.0)
index = np.argmax(prediction)
predicted_class = class_labels[index]

plt.imshow(image_rgb)
plt.axis('off')
plt.title(f"Prediction: {predicted_class}", fontsize=16)
print("Prediction is:", predicted_class)
plt.show()
