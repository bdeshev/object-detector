
import numpy as np
import gradio as gr

def sepia(input_img, text, slider_value):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


demo = gr.Interface(
    fn = sepia,
    title = "Image Filter",
    inputs = [
        gr.Image(label="Input Image"),
        gr.Textbox(label="Testing", value="Enter text here"),
        gr.Slider(label="Intensity", value=1, minimum=0, maximum=2, step=.2)
    ],
    outputs = "image"
)


demo.launch()