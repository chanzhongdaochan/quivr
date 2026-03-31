import tempfile
from uuid import uuid4

import chainlit as cl
from quivr_core import Brain
from quivr_core.llm import LLMEndpoint
from quivr_core.rag.entities.config import (
    DefaultModelSuppliers,
    LLMEndpointConfig,
    RetrievalConfig,
)
from quivr_core.embeddings import HashingEmbeddings


@cl.on_chat_start
async def on_chat_start():
    # 绕过 Chainlit 的文件持久化上传问题：直接让用户粘贴文本作为文档内容
    cl.user_session.set("brain", None)
    await cl.Message(
        content="请先把要提问的文本内容粘贴到第一条消息里（UTF-8 文本）。处理完成后我会提示你开始提问。"
    ).send()


def build_llm() -> LLMEndpoint:
    # DeepSeek is accessed via the OpenAI-compatible API.
    # You should set DEEPSEEK_API_KEY in your environment.
    return LLMEndpoint.from_config(
        LLMEndpointConfig(
            supplier=DefaultModelSuppliers.OPENAI,  # DeepSeek 使用 OpenAI 兼容接口
            model="deepseek-chat",
            llm_base_url="https://api.deepseek.com/v1",
            env_variable_name="DEEPSEEK_API_KEY",
            temperature=0.3,
        )
    )


def build_embedder() -> HashingEmbeddings:
    # Offline fallback embedding (no network dependency).
    return HashingEmbeddings()


@cl.on_message
async def main(message: cl.Message):
    brain = cl.user_session.get("brain")  # type: Brain | None

    # 第一次：把 message.content 当作“文档”，创建 brain
    if brain is None:
        content = message.content or ""
        if not content.strip():
            await cl.Message(content="文档内容为空，请重新发送一条包含文本的消息。").send()
            return

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name

        llm = build_llm()
        embedder = build_embedder()
        brain = Brain.from_files(
            name="user_brain",
            file_paths=[temp_file_path],
            llm=llm,
            embedder=embedder,
        )
        cl.user_session.set("brain", brain)
        cl.user_session.set("llm_config", llm.get_config())

        await cl.Message(content="文档处理完成。现在请在下一条消息里提问。").send()
        return

    # 后续：message.content 当作“问题”，走 RAG 流式回答
    path_config = "basic_rag_workflow.yaml"
    retrieval_config = RetrievalConfig.from_yaml(path_config)
    # Prevent yaml llm_config (default openai) from overriding the brain's DeepSeek llm.
    llm_config = cl.user_session.get("llm_config")
    if llm_config is not None:
        retrieval_config.llm_config = llm_config

    msg = cl.Message(content="", elements=[])
    await msg.send()

    saved_sources = set()
    saved_sources_complete = []
    elements = []

    async for chunk in brain.ask_streaming(
        question=message.content,
        run_id=uuid4(),
        retrieval_config=retrieval_config,
    ):
        await msg.stream_token(chunk.answer)
        for source in chunk.metadata.sources:
            if source.page_content not in saved_sources:
                saved_sources.add(source.page_content)
                saved_sources_complete.append(source)
                elements.append(
                    cl.Text(
                        name=source.metadata.get("original_file_name", "source"),
                        content=source.page_content,
                        display="side",
                    )
                )

    sources = "".join(
        f"- {source.metadata.get('original_file_name', 'source')}\n"
        for source in saved_sources_complete
    )
    msg.elements = elements
    msg.content = msg.content + f"\n\nSources:\n{sources}"
    await msg.update()
