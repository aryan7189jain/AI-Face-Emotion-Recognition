import cv2

from face_detector import detect_faces
from preprocess import preprocess_face
from predict import predict_emotion


# BGR colors per emotion (used for box + label background)
EMOTION_COLORS = {
    "Happy":    (60, 200, 60),
    "Sad":      (200, 130, 30),
    "Angry":    (40, 40, 220),
    "Fear":     (180, 30, 160),
    "Disgust":  (30, 140, 140),
    "Surprise": (30, 165, 240),
    "Neutral":  (160, 160, 160),
}

DEFAULT_COLOR = (0, 210, 255)


def _draw_annotation(image, box, emotion, confidence):
    """Draw a clean, premium-looking bounding box + label on the image."""

    x1, y1, x2, y2 = box
    color = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)

    # Bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Corner accents for a more "premium" tracked look
    corner_len = max(10, int((x2 - x1) * 0.15))
    thickness = 3
    for (cx, cy, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(image, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
        cv2.line(image, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

    # Label background + text
    label = f"{emotion}  {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    pad = 6
    label_y2 = y1
    label_y1 = max(0, y1 - th - baseline - 2 * pad)
    label_x1 = x1
    label_x2 = min(image.shape[1], x1 + tw + 2 * pad)

    overlay = image.copy()
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    cv2.putText(
        image,
        label,
        (label_x1 + pad, label_y2 - pad),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def analyze_image(image):

    faces = detect_faces(image)

    if len(faces) == 0:
        return image, []

    results = []

    for detected in faces:

        face = detected["face"]

        x1, y1, x2, y2 = detected["box"]

        processed_face = preprocess_face(face)

        emotion, confidence, all_probabilities = predict_emotion(processed_face)

        _draw_annotation(image, (x1, y1, x2, y2), emotion, confidence)

        results.append({
            "emotion": emotion,
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
            "probabilities": all_probabilities,
        })

    return image, results
