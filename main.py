
import tensorflow as tf
import numpy as np
import cv2

# loading the model
loaded_model = tf.keras.models.load_model('emotion_detection_model.h5')

# setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x = None
y = None

# defining labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initializing Classifier and VideoCapture
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fsd = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Loop runs till user presses "q"
while True:

    # Reads Images
    ret, full_size_image = fsd.read()

    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)

    faces = face.detectMultiScale(gray, 1.3, 10)

    # detecting faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # predicting the emotion
        yhat = loaded_model.predict(cropped_img)
        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                    1, cv2.LINE_AA)

        print("Emotion: " + labels[int(np.argmax(yhat))])

    # Displays Image
    cv2.imshow('Emotion', full_size_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break