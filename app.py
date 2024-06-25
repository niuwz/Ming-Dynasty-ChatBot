import os

import gradio as gr
import nltk
import torch
# from chatglm_llm import ChatGLM
from duckduckgo_search import ddg
from duckduckgo_search.utils import SESSION
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatZhipuAI
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


nltk.data.path.append('../nltk_data')

DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"



def search_web(query):

    SESSION.proxies = {
        "http": f"socks5h://localhost:7890",
        "https": f"socks5h://localhost:7890"
    }
    results = ddg(query)
    web_content = ''
    if results:
        for result in results:
            web_content += result['body']
    return web_content


def init_knowledge_vector_store(file):
    if file is None:
        return
    filepath = file.name
    embedding = HuggingFaceBgeEmbeddings(model_name="./bge_embedding/")
    if filepath is not None:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
        Chroma.from_documents(
            docs, embedding, persist_directory="./vector_store")


def get_knowledge_vector_store():
    embedding = HuggingFaceBgeEmbeddings(model_name="./bge_embedding/")
    vector_store = Chroma(embedding_function=embedding,
                          persist_directory="./vector_store")
    return vector_store


def get_knowledge_based_answer(
    query,
    large_language_model,
    vector_store,
    VECTOR_SEARCH_TOP_K,
    web_content,
    chat_history=[],
    history_len=3,
    temperature=0.01,
    top_p=0.9,
    key=None,
):
    if web_content:
        prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                            已知网络检索内容：{web_content}""" + """
                            已知内容:
                            {context}
                            问题:
                            {question}"""
    else:
        prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

            已知内容:
            {context}

            问题:
            {question}"""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    glm = ChatZhipuAI(model=large_language_model, temperature=temperature,
                      top_p=top_p, api_key=key)

    knowledge_chain = RetrievalQA.from_llm(
        llm=glm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")

    knowledge_chain.return_source_documents = True
    result = knowledge_chain.invoke({"query": query})

    return result['result']


def clear_session():
    return '', None


def predict(input,
            large_language_model,
            VECTOR_SEARCH_TOP_K,
            # history_len,
            temperature,
            top_p,
            use_web,
            key,
            history=None):
    if history == None:
        history = []
    # if file_obj is not None:
    #     print(file_obj.name)
    #     used.value
    #     vector_store = init_knowledge_vector_store(file_obj.name)
    # else:
    #     vector_store = init_knowledge_vector_store(None)
    vector_store = get_knowledge_vector_store()
    if use_web == 'True':
        web_content = search_web(query=input)
        if web_content is None:
            web_content = ""
    else:
        web_content = ''

    resp = get_knowledge_based_answer(
        query=input,
        large_language_model=large_language_model,
        vector_store=vector_store,
        VECTOR_SEARCH_TOP_K=VECTOR_SEARCH_TOP_K,
        web_content=web_content,
        chat_history=history,
        # history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        key=key
    )
    print(resp)
    history.append((input, resp))
    return '', history, history


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("""<h1><center>明史问答小助手</center></h1>""")
        with gr.Row():
            with gr.Column(scale=1):
                model_choose = gr.Accordion("模型选择")
                with model_choose:
                    large_language_model = gr.Dropdown(
                        ["GLM-3-turbo", "GLM-4-0520", 'GLM-4'],
                        label="large language model",
                        value="GLM-4")

                    key = gr.Textbox(label="请输入您的API Key", type='password')

                file_load = gr.Accordion("知识库配置")

                with file_load:
                    know = gr.Tab(label="预定义知识库")
                    load = gr.Tab(label="上传知识库文件")
                    with know:
                        # used = gr.Markdown(
                        #     value="""- 明史简述\n - 朱元璋传\n - 明朝那些事\n""")
                        used = gr.HTML(
                            value="<li>明史简述</li><li>朱元璋传</li><li>明朝那些事</li>")
                    with load:
                        file = gr.File(label='请上传txt,md,docx类型文件',
                                       file_types=['.txt', '.md', '.docx'],
                                       )
                        get_vs = gr.Button("生成知识库")
                        get_vs.click(init_knowledge_vector_store,
                                     inputs=[file])

                use_web = gr.Radio(["True", "False"],
                                   label="Web Search",
                                   value="False")
                model_argument = gr.Accordion("生成参数配置")

                with model_argument:

                    VECTOR_SEARCH_TOP_K = gr.Slider(
                        1,
                        10,
                        value=6,
                        step=1,
                        label="vector search top k",
                        interactive=True)

                    # HISTORY_LEN = gr.Slider(0,
                    #                         3,
                    #                         value=0,
                    #                         step=1,
                    #                         label="history len",
                    #                         interactive=True)
                    # TODO:history
                    # HISTORY_LEN = None

                    temperature = gr.Slider(0,
                                            1,
                                            value=0.8,
                                            step=0.01,
                                            label="temperature",
                                            interactive=True)
                    top_p = gr.Slider(0,
                                      1,
                                      value=0.9,
                                      step=0.1,
                                      label="top_p",
                                      interactive=True)

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label='明史知识问答助手', height=600)
                message = gr.Textbox(label='请输入您的问题')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")
                    send.click(predict,
                               inputs=[
                                   message, large_language_model,  VECTOR_SEARCH_TOP_K,
                                 temperature, top_p, use_web, key,
                                   state
                               ],
                               outputs=[message, chatbot, state])
                    clear_history.click(fn=clear_session,
                                        inputs=[],
                                        outputs=[chatbot, state],
                                        queue=False)

                    message.submit(predict,
                                   inputs=[
                                       message, large_language_model, VECTOR_SEARCH_TOP_K, 
                                       temperature, top_p, use_web, key, state
                                   ],
                                   outputs=[message, chatbot, state])

        gr.Markdown("""<center><font size=2>
        本项目基于LangChain和GLM系列模型, 提供基于本地知识的自动问答应用. <br>
        主要参考了项目<a href='https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui' target="_blank">LangChain-ChatGLM-Webui</a>
        目前项目知识数据库来源较为单一, 可通过上传自定义文件等作为补充. <br>
        后续将提供更多的模型以及更加多样化的数据来源, 欢迎关注<a href='https://github.com/niuwz/Ming-Dynasty-ChatBot' target="_blank">Github地址</a>.
        </center></font>
        """)
    demo.queue().launch(share=False)
