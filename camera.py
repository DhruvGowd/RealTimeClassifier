import numpy as np
import cv2
import predict
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing import image

org    = (50,50)
thicc  = 2
color  = (255, 0, 0)
scale  = 1
font   = cv2.FONT_HERSHEY_SIMPLEX
width  = 150
height = 150
model  = tf.keras.models.load_model('model')

def getPrediction(img):
    x = np.expand_dims(img, axis=0)
    images = np.vstack([x])
    return model.predict(images)[0][0]

def runFeed():

    cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) & 0xFF != ord('q'):
        # Capture frame-by-frame
        ret, frame = cap.read()
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        faces = faceCascade.detectMultiScale(
                    frame,
                    scaleFactor=1.5,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

        sample = cv2.resize(frame, dsize=(width,height))

        if len(faces) > 0:
            who = predict.getPrediction(sample)

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                if who > 0.5:
                    cv2.putText(frame, 'Dhruv', org, font,
                       scale, color, thicc, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'FATTY', org, font,
                       scale, color, thicc, cv2.LINE_AA)
            cv2.imshow('frame',frame)
        else:
            cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    runFeed()
