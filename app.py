import gradio as gr
import spaces
import tensorflow as tf
import numpy as np

@spaces.GPU
def tensorflow_test():
    # Verify TensorFlow can initialize and run a tiny CPU operation
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.matmul(x, x)
    return f"✅ TensorFlow works. Result: {y.numpy().tolist()}"

with gr.Blocks(title="TensorFlow ZeroGPU Test") as demo:
    gr.Markdown("# 🧪 TensorFlow + ZeroGPU Test")
    gr.Markdown("This test loads TensorFlow but does not load Ultralytics or any project model.")
    btn = gr.Button("Test TensorFlow", variant="primary")
    result = gr.Markdown("Waiting for test...")
    btn.click(fn=tensorflow_test, inputs=[], outputs=result)

if __name__ == "__main__":
    demo.launch()
