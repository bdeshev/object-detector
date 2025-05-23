import gradio as gr
import numpy as np

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img @ sepia_filter.T
    sepia_img = np.clip(sepia_img, 0, 255)
    sepia_img = sepia_img.astype(np.uint8)
    return sepia_img

with gr.Blocks() as demo:
    gr.Markdown("# Live Sepia Webcam")
    with gr.Row():
        webcam = gr.Image(sources="webcam", streaming=True)
        output = gr.Image()
    
    webcam.stream(
        fn=sepia,
        inputs=webcam,
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()