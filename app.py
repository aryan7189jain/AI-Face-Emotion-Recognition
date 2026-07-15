import time

import gradio as gr
import cv2

from emotion_analyzer import analyze_image, EMOTION_COLORS


EMOTION_ICONS = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fear": "😨",
    "Disgust": "🤢",
    "Surprise": "😲",
    "Neutral": "😐",
}

# Hex versions of the BGR colors defined in emotion_analyzer, for HTML cards
EMOTION_HEX = {
    name: "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])
    for name, bgr in EMOTION_COLORS.items()
}
DEFAULT_HEX = "#00d2ff"


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _stat_card(icon, label, value, accent="#00d2ff"):
    return f"""
    <div class="stat-card" style="--accent:{accent}">
        <div class="stat-icon">{icon}</div>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """


def _empty_cards():
    return (
        _stat_card("👥", "Faces Detected", "0"),
        _stat_card("😊", "Dominant Emotion", "—"),
        _stat_card("🎯", "Confidence", "—"),
    )


def _build_table(results):
    table = []
    for i, result in enumerate(results):
        icon = EMOTION_ICONS.get(result["emotion"], "")
        table.append([
            f"Face {i + 1}",
            f"{icon} {result['emotion']}",
            f"{result['confidence'] * 100:.2f}%",
        ])
    return table


# ---------------------------------------------------------------------------
# Static image analysis
# ---------------------------------------------------------------------------

def predict(image):

    if image is None:
        faces_card, emotion_card, confidence_card = _empty_cards()
        return None, None, faces_card, emotion_card, confidence_card, []

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    output_image, results = analyze_image(bgr_image)

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    save_path = "output.png"
    cv2.imwrite(save_path, output_image)

    if len(results) == 0:
        faces_card = _stat_card("👥", "Faces Detected", "0")
        emotion_card = _stat_card("😊", "Dominant Emotion", "None")
        confidence_card = _stat_card("🎯", "Confidence", "0%")
        return output_image_rgb, save_path, faces_card, emotion_card, confidence_card, []

    table = _build_table(results)

    num_faces = len(results)
    best = max(results, key=lambda x: x["confidence"])
    accent = EMOTION_HEX.get(best["emotion"], DEFAULT_HEX)

    faces_card = _stat_card("👥", "Faces Detected", str(num_faces))
    emotion_card = _stat_card(
        EMOTION_ICONS.get(best["emotion"], "😊"),
        "Dominant Emotion",
        best["emotion"],
        accent,
    )
    confidence_card = _stat_card(
        "🎯", "Highest Confidence", f"{best['confidence'] * 100:.2f}%", accent
    )

    return output_image_rgb, save_path, faces_card, emotion_card, confidence_card, table


# ---------------------------------------------------------------------------
# Live webcam analysis (streaming)
# ---------------------------------------------------------------------------

def process_webcam_frame(frame):

    if frame is None:
        faces_card, emotion_card, confidence_card = _empty_cards()
        return None, faces_card, emotion_card, confidence_card, []

    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    output_frame, results = analyze_image(bgr_frame)

    output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    if len(results) == 0:
        faces_card = _stat_card("👥", "Faces Detected", "0")
        emotion_card = _stat_card("😊", "Dominant Emotion", "None")
        confidence_card = _stat_card("🎯", "Confidence", "0%")
        return output_frame_rgb, faces_card, emotion_card, confidence_card, []

    table = _build_table(results)
    num_faces = len(results)
    best = max(results, key=lambda x: x["confidence"])
    accent = EMOTION_HEX.get(best["emotion"], DEFAULT_HEX)

    faces_card = _stat_card("👥", "Faces Detected", str(num_faces))
    emotion_card = _stat_card(
        EMOTION_ICONS.get(best["emotion"], "😊"),
        "Dominant Emotion",
        best["emotion"],
        accent,
    )
    confidence_card = _stat_card(
        "🎯", "Highest Confidence", f"{best['confidence'] * 100:.2f}%", accent
    )

    return output_frame_rgb, faces_card, emotion_card, confidence_card, table


# ---------------------------------------------------------------------------
# Theme + CSS
# ---------------------------------------------------------------------------

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

.gradio-container {
    max-width: 1240px !important;
    margin: auto;
    background: radial-gradient(circle at top, #16213e 0%, #0b0f1a 55%, #05070d 100%) !important;
    font-family: 'Poppins', sans-serif !important;
}

footer { display: none !important; }

/* ---------- Header ---------- */
#app-header {
    text-align: center;
    padding: 36px 12px 18px 12px;
}
#app-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.5px;
}
#app-header .subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 6px;
    font-weight: 400;
}
#app-header .badges {
    margin-top: 14px;
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}
#app-header .badge {
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #cbd5e1;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
}

/* ---------- Generic panels ---------- */
.block, .form {
    border-radius: 20px !important;
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
}

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* ---------- Buttons ---------- */
.gr-button, button.primary {
    height: 54px;
    font-size: 17px !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    border: none !important;
    box-shadow: 0 8px 22px rgba(124,58,237,0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.gr-button:hover, button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(124,58,237,0.5) !important;
}

/* ---------- Stat cards ---------- */
.stat-card {
    --accent: #00d2ff;
    border-radius: 18px;
    padding: 18px 16px;
    text-align: center;
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 3px solid var(--accent);
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
}
.stat-icon { font-size: 1.8rem; margin-bottom: 4px; }
.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 4px;
}

/* ---------- Media + tables ---------- */
.gr-image, img { border-radius: 16px !important; }
table { border-radius: 12px !important; overflow: hidden; }

/* ---------- Tabs ---------- */
.tab-nav button {
    font-weight: 600 !important;
    color: #cbd5e1 !important;
}
.tab-nav button.selected {
    color: #ffffff !important;
    border-bottom: 3px solid #7c3aed !important;
}
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(theme=theme, css=custom_css, title="AI Face Emotion Detector") as demo:

    gr.HTML(
        """
        <div id="app-header">
            <h1>✨ AI Face Emotion Detector</h1>
            <div class="subtitle">Real-time human emotion recognition powered by deep learning</div>
            <div class="badges">
                <span class="badge">🧠 TensorFlow CNN</span>
                <span class="badge">👁️ OpenCV YuNet</span>
                <span class="badge">⚡ Live Webcam</span>
                <span class="badge">🎛️ Gradio UI</span>
            </div>
        </div>
        """
    )

    with gr.Tabs():

        # ================= IMAGE UPLOAD TAB =================
        with gr.Tab("📤 Upload Image"):

            with gr.Row(equal_height=True):

                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    input_image = gr.Image(type="numpy", label="Upload an Image", sources=["upload", "clipboard"])
                    analyze_btn = gr.Button("🔍 Analyze Emotion", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Result")
                    output_image = gr.Image(type="numpy", label="Detected Faces")
                    download_image = gr.File(label="📥 Download Result")

            gr.Markdown("---")
            gr.Markdown("### 📊 Analysis Summary")

            with gr.Row():
                faces_card = gr.HTML(_stat_card("👥", "Faces Detected", "0"))
                emotion_card = gr.HTML(_stat_card("😊", "Dominant Emotion", "—"))
                confidence_card = gr.HTML(_stat_card("🎯", "Confidence", "—"))

            gr.Markdown("### 👥 Face Details")
            prediction = gr.Dataframe(
                headers=["Face", "Emotion", "Confidence"],
                datatype=["str", "str", "str"],
                interactive=False,
                label="Detected Faces",
            )

            analyze_btn.click(
                fn=predict,
                inputs=input_image,
                outputs=[
                    output_image,
                    download_image,
                    faces_card,
                    emotion_card,
                    confidence_card,
                    prediction,
                ],
            )

        # ================= LIVE WEBCAM TAB =================
        with gr.Tab("🎥 Live Webcam"):

            gr.Markdown(
                "### Real-time detection\n"
                "Enable your webcam below — each frame is analyzed live and annotated automatically."
            )

            with gr.Row(equal_height=True):

                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                        label="Webcam",
                    )

                with gr.Column(scale=1):
                    webcam_output = gr.Image(type="numpy", label="Live Detection")

            gr.Markdown("### 📊 Live Summary")

            with gr.Row():
                live_faces_card = gr.HTML(_stat_card("👥", "Faces Detected", "0"))
                live_emotion_card = gr.HTML(_stat_card("😊", "Dominant Emotion", "—"))
                live_confidence_card = gr.HTML(_stat_card("🎯", "Confidence", "—"))

            live_table = gr.Dataframe(
                headers=["Face", "Emotion", "Confidence"],
                datatype=["str", "str", "str"],
                interactive=False,
                label="Live Detected Faces",
            )

            webcam_input.stream(
                fn=process_webcam_frame,
                inputs=webcam_input,
                outputs=[
                    webcam_output,
                    live_faces_card,
                    live_emotion_card,
                    live_confidence_card,
                    live_table,
                ],
                stream_every=0.3,
                time_limit=None,
            )

    gr.HTML(
        '<div style="text-align:center; color:#64748b; padding: 24px 0 8px 0; font-size:0.85rem;">'
        "Built with TensorFlow · OpenCV YuNet · Gradio"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch()
