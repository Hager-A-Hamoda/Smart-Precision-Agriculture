import gradio as gr
import spaces
from ultralytics import YOLO

@spaces.GPU
def ultralytics_test():
    # Import only: no project weights/model files are loaded.
    return "✅ Ultralytics import works inside ZeroGPU."

with gr.Blocks(title="Ultralytics ZeroGPU Test") as demo:
    gr.Markdown("# 🧪 Ultralytics + ZeroGPU Test")
    gr.Markdown("This test loads Ultralytics only. No TensorFlow and no project model.")
    btn = gr.Button("Test Ultralytics", variant="primary")
    result = gr.Markdown("Waiting for test...")
    btn.click(fn=ultralytics_test, inputs=[], outputs=result)

if __name__ == "__main__":
    demo.launch()
