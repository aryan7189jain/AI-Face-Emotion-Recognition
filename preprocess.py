import cv2
import numpy as np

def preprocess_face(face):

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(face, (48,48))

    face = face.astype("float32") / 255.0

    face = np.expand_dims(face, axis=-1)

    return face