# -*- encoding: utf-8 -*-
"""
########################################
@File       :   run_gradio.py
@CreateTime :   2024/02/04 20:33:34
@ModifyTime :   2024/02/04 20:33:34
@Author     :   Shixian Ning 
@Version    :   1.0
@Contact    :   ningshixian@lixiang.com
@License    :   (C)Copyright 2021-2025, zxw
@Desc       :   None
########################################
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import uniqa.api.gradio as gr
import uuid
import time
from uniqa.components.indexs.faiss_index import FaissEmbeddingSearcher
from configs.config import qa_pair_path

import torch
import atexit
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"


# 程序退出时执行清理
@atexit.register
def clean():
    """清理内存。这个函数会清理PyTorch的GPU缓存，以及触发Python的垃圾回收。"""
    try:
        # 检查是否有可用的CUDA设备
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"清理内存时出错: {e}")


from uniqa.components.indexs.faiss_index import FaissEmbeddingSearcher
from configs.config import *
model_type = "stella-large"
# pretrained_model_path = pretrained_model_config[model_type]
save_ft_model_path, npy_path = fine_tuned_model_config[model_type]

searcher = FaissEmbeddingSearcher(
    model_path=save_ft_model_path,
    qa_path=qa_pair_path,
    save_npy_path=npy_path
)
searcher.main("初始化索引", 5)


if __name__ == "__main__":

    with gr.Blocks() as demo:
        session = gr.State(uuid.uuid4)
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown(
                    """<h1><center>Call Center faq问答</center></h1>
                    <center> LLM 采用通义千问</center>
                    """
                )
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=450, show_copy_button=True)
                msg = gr.Textbox(label="Query/问题")

                with gr.Row():
                    # 创建提交按钮。
                    db_wo_his_btn = gr.Button("Chat")
                with gr.Row():
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(components=[chatbot], value="Clear console")

            def user(user_message, history):
                return "", history + [[user_message, ""]]

            def func(history):
                closest_matches = searcher.search(history[-1][0])
                answer = "\n".join(["\t".join(list(map(str, item.values()))) for item in closest_matches])
                for char in answer:
                    history[-1][1] += char
                    # time.sleep(0.05)
                    yield history

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                func, [chatbot], chatbot
            )

            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            # db_wo_his_btn.click(li_bot.get_response, inputs=[
            #                 msg], outputs=[msg, chatbot])
            db_wo_his_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                func, inputs=[chatbot], outputs=chatbot
            )

        gr.Markdown(
            """提醒：<br>
        1. 初始化数据库时间可能较长，请耐心等待。
        2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
        """
        )
    # threads to consume the request
    gr.close_all()
    # 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
    # demo.launch(share=True, server_port=7861)
    demo.queue().launch(
        server_name="0.0.0.0", server_port=9000, share=False, inbrowser=True
    )
