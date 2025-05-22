import gradio as gr
import numpy as np

def flip_vertically(image):
    if image is not None:
        return np.flip(image, axis=0)
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Live Video Flipper")
    with gr.Row():
        webcam = gr.Image(sources="webcam", streaming=True)
        output = gr.Image()
    
    webcam.change(
        flip_vertically,
        inputs=webcam,
        outputs=output,
        show_progress=False
    )

if __name__ == "__main__":
    demo.launch()