import gradio as gr
import spaces

@spaces.GPU
def tensorflow_lazy_test():
    import tensorflow as tf
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.matmul(x, x)
    return f"✅ TensorFlow lazy import works: {y.numpy().tolist()}"

@spaces.GPU
def ultralytics_lazy_test():
    from ultralytics import YOLO
    return "✅ Ultralytics lazy import works."

with gr.Blocks(title="Lazy Import ZeroGPU Test") as demo:
    gr.Markdown("# 🧪 Lazy Import ZeroGPU Test")
    gr.Markdown(
        "TensorFlow and Ultralytics are NOT imported at startup. "
        "Each is imported only inside its own @spaces.GPU function."
    )

    tf_btn = gr.Button("Test TensorFlow", variant="primary")
    tf_result = gr.Markdown("TensorFlow test not run.")
    tf_btn.click(fn=tensorflow_lazy_test, inputs=[], outputs=tf_result)

    yolo_btn = gr.Button("Test Ultralytics", variant="primary")
    yolo_result = gr.Markdown("Ultralytics test not run.")
    yolo_btn.click(fn=ultralytics_lazy_test, inputs=[], outputs=yolo_result)

if __name__ == "__main__":
    demo.launch()
