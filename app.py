import os
import logging
import gradio as gr
from codeine.frontend.utils import convert_mdtext, reset_textbox, transfer_input, reset_state, delete_last_conversation, cancel_outputing, convert_to_markdown
from codeine.frontend.presets import small_and_beautiful_theme, title, description_top, description
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

# Import the chat_engine object
#from llama_index.chat_engine import SimpleChatEngine, ReActChatEngine
#from langchain.chat_models import ChatOpenAI
#from llama_index import ServiceContext
#service_context = ServiceContext.from_defaults(llm=ChatOpenAI(temperature=0.))
#chat_engine = SimpleChatEngine.from_defaults(service_context=service_context)
from codeine.chatbot import chat_engine

with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()


total_count = 0
def predict(text, chatbot, history, top_p, temperature, max_length_tokens, max_context_length_tokens):
    if text == "":
        yield chatbot, history, "Empty context."
        return

    global total_count
    total_count += 1
    print(total_count)

    response = chat_engine.chat(text)
    
    generated_text = response.response
    # TODO: ignoring these for now, but could be useful
    #source_nodes: List[NodeWithScore] = field(default_factory=list)
    #extra_info: Optional[Dict[str, Any]] = None
    # TODO: streaming not implemented yet
    #generated_text = ""
    #for token in response:
    #    # Process the output token by token
    #    generated_text += token

    # Update the chatbot and history states
    chatbot_updated = chatbot + [[text, convert_mdtext(generated_text)]]
    history_updated = history + [[text, generated_text]]

    yield chatbot_updated, history_updated, "Generating..."

    try:
        yield chatbot_updated, history_updated, "Generate: Success"
    except:
        pass


with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    history = gr.State([])
    user_question = gr.State("")
    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")
    gr.Markdown(description_top)
    with gr.Row(scale=1).style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row(scale=1):
                chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height="100%")
            with gr.Row(scale=1):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Enter text"
                    ).style(container=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("Send")
                with gr.Column(min_width=70, scale=1):
                    cancelBtn = gr.Button("Stop")
            with gr.Row(scale=1):
                emptyBtn = gr.Button(
                    "🧹 New Conversation",
                )
                retryBtn = gr.Button("🔄 Regenerate")
                delLastBtn = gr.Button("🗑️ Remove Last Turn") 
        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Parameters")
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=512,
                        value=512,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )
    gr.Markdown(description)

    predict_args = dict(
        fn=predict,
        inputs=[
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    reset_args = dict(
        fn=reset_textbox, inputs=[], outputs=[user_input, status_display]
    )

    transfer_input_args = dict(
        fn=transfer_input, inputs=[user_input], outputs=[user_question, user_input, submitBtn], show_progress=True
    )

    predict_event1 = user_input.submit(**transfer_input_args).then(**predict_args)

    predict_event2 = submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )
    cancelBtn.click(
        cancel_outputing, [], [status_display], 
        cancels=[
            predict_event1, predict_event2
        ]
    )    
demo.title = "Baize"

demo.queue(concurrency_count=1).launch()
