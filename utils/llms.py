from langchain_openai import ChatOpenAI
# from langchain_community.chat_models.openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
from typing import Callable, Awaitable

# 加载环境变量
# 1. 先尝试从当前目录加载.env文件
# 2. 如果找不到,则尝试从父目录递归查找
print("加载环境变量...")
dotenv_path = find_dotenv(raise_error_if_not_found=True)
load_dotenv(dotenv_path, override=True)

# 获取必需的环境变量并进行验证
print(os.getenv("OPENAI_LLM_BASE_URL"))
print(os.getenv("OPENAI_LLM_MODEL"))
print(os.getenv("LOCAL_LLM_BASE_URL"))
print(os.getenv("LOCAL_LLM_MODEL"))

DEFAULT_MODEL = "gpt-4o-mini"


# 创建本地 LLM 实例
local_llm = ChatOpenAI(
    api_key="EMPTY",     # type: ignore
    base_url=os.getenv("LOCAL_LLM_BASE_URL"),
    model=os.getenv("LOCAL_LLM_MODEL") or DEFAULT_MODEL
)
openai_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_LLM_API_KEY"), # type: ignore
    base_url=os.getenv("OPENAI_LLM_BASE_URL"),
    model=os.getenv("OPENAI_LLM_MODEL"), # type: ignore
    temperature=0.3,
)


# 创建 OpenAI LLM 实例
# 注意: 由于 OpenAI 的限制,需要显式设置 OPENAI_API_KEY 环境变量
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_LLM_API_KEY")
# import openai

# openai.base_url = os.getenv("OPENAI_LLM_BASE_URL")
# openai.api_key = os.getenv("OPENAI_LLM_API_KEY")

# openai_llm_naive = openai.Client(
#     base_url=os.getenv("OPENAI_LLM_BASE_URL"),
#     api_key=os.getenv("OPENAI_LLM_API_KEY"),
#     timeout=30.0
# )

# import openai
# def create__llm(base_url: str, api_key: str, model: str, stop: list[str] | None = None)

def create_llm_invoker(llm: ChatOpenAI, stop: list[str] | None = None) -> Callable[[str], Awaitable[str]]:
    async def ainvoke(prompt: str) -> str:
        if stop is None:
            return (await llm.ainvoke(prompt)).content # type: ignore
        else:
            content = ""
            # 🔄 修复: 移除多余的 await，因为 astream() 已经返回了一个 AsyncIterator
            async for chunk in llm.astream(prompt):
                # 🛑 检查是否包含停止词
                for stop_token in stop:
                    if stop_token in chunk.content:
                        return content
                content += chunk.content # type: ignore
            return content
    return ainvoke

def chat_completion(prompt: str, model: str = os.getenv("OPENAI_LLM_MODEL") or DEFAULT_MODEL) -> str:
    """调用 OpenAI API 进行对话补全

    参数:
        prompt: 输入提示词
        model: 模型名称,默认从环境变量获取

    返回:
        str: 模型的回复内容
    """
    try:
        import requests

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_LLM_API_KEY')}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(
            f"{os.getenv('OPENAI_LLM_BASE_URL')}/chat/completions",
            headers=headers,
            json=data,
            timeout=30.0
        )

        if response.status_code == 200:
            print(response.json())
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"API 请求失败: {response.status_code}")
            return ""

    except Exception as e:
        print(f"OpenAI API 调用出错: {str(e)}")
        return ""



if __name__ == "__main__":
    # print("invoke openai_llm")
    # print(openai_llm.openai_api_base, openai_llm.openai_api_key, openai_llm.model_name)
    # print(chat_completion("Hello, world!"))
    # 测试异步调用
    # import asyncio
    # from langchain_openai import ChatOpenAI

    # async def test_ainvoke():
    #     """🧪 测试异步调用"""
    #     response = await ainvoke("你好,请用中文回答:1+1等于几?请你用两行回答", llm=openai_llm, stop=["\n"])
    #     print(f"🤖 异步调用结果: {response}")

    # # 运行异步测试
    # asyncio.run(test_ainvoke())



    pass
