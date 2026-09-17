import gradio as gr
import spaces

@spaces.GPU
def gpu_test():
    return "✅ ZeroGPU + @spaces.GPU is working."

with gr.Blocks(title="Smart Precision Agriculture - ZeroGPU Test") as demo:
    gr.Markdown("# 🌱 ZeroGPU Diagnostic Test")
    gr.Markdown("This test loads only Gradio + spaces.")
    btn = gr.Button("Test ZeroGPU", variant="primary")
    result = gr.Markdown("Waiting for test...")
    btn.click(fn=gpu_test, inputs=[], outputs=result)

if __name__ == "__main__":
    demo.launch()
