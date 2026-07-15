import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_detection_yunet_2023mar.onnx"
)

face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)


def detect_faces(image):
    """
    Detect faces using OpenCV YuNet.

    Parameters:
        image : BGR image

    Returns:
        List of dictionaries
    """

    h, w = image.shape[:2]

    # Tell YuNet the image size
    face_detector.setInputSize((w, h))

    _, detections = face_detector.detect(image)

    faces = []

    if detections is None:
        return faces

    for detection in detections:

        x, y, width, height = detection[:4].astype(int)

        confidence = float(detection[-1])

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        faces.append(
            {
                "face": face,
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
            }
        )

    return faces