import gradio as gr

def health_check():
    return "✅ Gradio/ZeroGPU startup test passed. No TensorFlow or Ultralytics is loaded."

with gr.Blocks(title="Smart Precision Agriculture - Startup Test") as demo:
    gr.Markdown("# 🌱 Smart Precision Agriculture")
    gr.Markdown("## ZeroGPU startup diagnostic")
    test_btn = gr.Button("Run Startup Test", variant="primary")
    result = gr.Markdown("Click the button to test the Gradio runtime.")
    test_btn.click(fn=health_check, inputs=[], outputs=result)

if __name__ == "__main__":
    demo.launch()
