import gradio as gr
import spaces
import tensorflow as tf
from ultralytics import YOLO

@spaces.GPU
def combined_test():
    # Import and execute a tiny operation from both libraries.
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.matmul(x, x)

    # Do not load any project weights.
    # Importing YOLO confirms the Ultralytics side is available.
    _ = YOLO

    return f"✅ TensorFlow + Ultralytics work together. TensorFlow result: {y.numpy().tolist()}"

with gr.Blocks(title="TensorFlow + Ultralytics ZeroGPU Test") as demo:
    gr.Markdown("# 🧪 TensorFlow + Ultralytics + ZeroGPU Test")
    gr.Markdown("No project models or weights are loaded in this test.")
    btn = gr.Button("Test Both Libraries", variant="primary")
    result = gr.Markdown("Waiting for test...")
    btn.click(fn=combined_test, inputs=[], outputs=result)

if __name__ == "__main__":
    demo.launch()
