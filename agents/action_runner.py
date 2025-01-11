import os
os.environ["http_proxy"] = "http://172.31.226.127:7890"
os.environ["https_proxy"] = "http://172.31.226.127:7890"
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
# from langchain_community.tools.wikipedia import Wiki
from langchain_community.docstore.wikipedia import Wikipedia    # 这是一个docstore
from langchain.agents.react.base import DocstoreExplorer

# 🌐 设置代理环境变量以访问维基百科


def create_wikipedia_docstore() -> DocstoreExplorer:
    return DocstoreExplorer(docstore=Wikipedia())



# api_wrapper = WikipediaAPIWrapper(
#     top_k_results=1,
#     # doc_content_chars_max=500,
#     wiki_client=None
# )


# tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# async def search(query: str) -> str:
#     answer = await tool.ainvoke({"query": query})
#     return answer


# TODO: 将 Wikipedia 集成到系统中
