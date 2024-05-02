import gradio as gr
from models import FaceRecognition

def main():
    css = """
    h1 {
        text-align: center;
        display:block;
        }
    """
    demo = gr.Blocks(css=css)
    model = FaceRecognition()
    
    with demo:
        gr.Markdown("""
        # INEOM EAM AI Model Runtime by vixmo.ai
        """)
        
        with gr.Tab('Register'):
            with gr.Column():
                vid_source0 = gr.Video()
                with gr.Row():
                    name = gr.Text(label='Enter your name here', scale=3)
                    submit = gr.Button('Submit', scale=1)
            
        with gr.Tab('Login'):
            with gr.Column():
                img_source = gr.Image(type='filepath')
                with gr.Row():
                    login_result = gr.Text(label='', interactive=False, scale=3)
                    detect_btn = gr.Button('Submit', scale=1)
                
        submit.click(model.registration_faces,
                     inputs=[name, vid_source0],
                     outputs=[name])
        
        detect_btn.click(model.recognize_face,
                         inputs=[img_source],
                         outputs=[login_result])
        
    demo.launch(share=False, server_name='0.0.0.0')
    
if __name__ == '__main__':
    main()