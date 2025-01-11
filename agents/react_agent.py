from ast import Tuple
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
import re
# from agents.action_runner import search
from utils.fewshots import WEBTHINK_SIMPLE3
from rich import print
from rapidfuzz import fuzz
from typing import Awaitable, List, Tuple, Callable
from langchain.agents.react.base import DocstoreExplorer
# from langchain.docstore.base import Docstore
from langchain_community.docstore.wikipedia import Wikipedia
from agents.action_runner import create_wikipedia_docstore

class ReactAgentState(BaseModel):
    question: str       # 问题
    key: str            # 答案的标准或关键
    answer: str = ""    # 当前答案
    is_correct: bool | None = None # 是否正确
    step_n: int = 0     # 当前轮次
    finished: bool = False # 是否完成
    scratchpad: str = "" # 临时记录
    error: str | None = None # 错误信息  # 先前执行的错误
    need_check: bool = False # 是否需要检查答案
    previous_search_doc: str | None = None


async def run_react_agent(
    question: str,
    key: str,
    llm: Callable[[str], Awaitable[str]],
    check_llm: Callable[[str], Awaitable[str]] | None = None,
    agent_format_func: Callable[[ReactAgentState], str] = lambda x: format_agent(WEBTHINK_SIMPLE3, x.scratchpad, x.question)
) -> str:
    # 初始化状态
    state = ReactAgentState(question=question, key=key)
    docstore = create_wikipedia_docstore()
    print(f"[blue]📝 初始化状态: {state}[/blue]")
    while not state.finished:
        print("="*50)
        print(f"[blue]📝 进入循环[/blue]")
        state = await step_react_agent(state, llm, check_llm or llm, docstore=docstore, agent_format_func=agent_format_func)
        print("="*50)
        # print(f"[blue]📝 完成一轮: {state}[/blue]")
        # break
    return state.answer

async def step_react_agent(
    state: ReactAgentState,
    llm: Callable[[str], Awaitable[str]],
    check_llm: Callable[[str], Awaitable[str]],
    docstore: DocstoreExplorer,
    agent_format_func: Callable[[ReactAgentState], str]
) -> ReactAgentState:
    new_state = state.model_copy()
    new_state.step_n += 1

    print(f"[blue]📝 进入第 {new_state.step_n} 步[/blue]")

    try:
        # 🤔 执行思考-行动-观察循环

        await think(new_state, llm=llm, agent_format_func=agent_format_func)

        action = await act(new_state, llm=llm, agent_format_func=agent_format_func)

        observation, is_finish = await observe(new_state, action, llm, check_answer, check_llm=check_llm, docstore=docstore)

        # 📝 更新状态
        return new_state

    except ValueError as e:
        # ❌ 错误处理：使用原始状态重试
        print(f"[red]📝 动作执行错误: {e}[/red]")
        state.error = str(e)
        state.step_n += 1
        return await step_react_agent(state, llm, check_llm, docstore, agent_format_func)

# 🧠 新增的辅助函数
async def think(state: ReactAgentState, llm: Callable[[str], Awaitable[str]], agent_format_func: Callable[[ReactAgentState], str] = lambda x: format_agent(WEBTHINK_SIMPLE3, x.scratchpad, x.question)) -> str:
    """思考阶段：分析当前情况并形成想法"""
    state.scratchpad += f"\nThought {state.step_n}:"
    prompt = agent_format_func(state)
    # print(f"[blue]📝 Thought 输入: [italic]{prompt}[/italic][/blue]")
    thought = await llm(prompt +"\n(Note: Write down your thoughts in one line without Thought prefix.)")
    print(f"[green]📝 Thought 输出: {thought}[/green]")
    state.scratchpad += thought
    return thought

async def act(state: ReactAgentState, llm: Callable[[str], Awaitable[str]], agent_format_func: Callable[[ReactAgentState], str] = lambda x: format_agent(WEBTHINK_SIMPLE3, x.scratchpad, x.question)) -> str:
    """行动阶段：基于思考决定下一步行动"""
    state.scratchpad += f"\nAction {state.step_n}:"
    prompt = agent_format_func(state)
    # print(f"[blue]📝 Action 输入: [italic]{prompt}[/italic][/blue]")
    action = await llm(prompt)
    print(f"[green]📝 Action 输出: {action}[/green]")
    state.scratchpad += action
    return action

async def observe(state: ReactAgentState, action: str , llm: Callable[[str], Awaitable[str]], check_func: Callable[[str, str, str, Callable[[str], Awaitable[str]]], Awaitable[bool]], check_llm: Callable[[str], Awaitable[str]], docstore: DocstoreExplorer) -> Tuple[str, bool]:
    """观察阶段：执行行动并观察结果"""

    action_type, argument = parse_action(action)
    state.scratchpad += f"\nObservation {state.step_n}:"

    observation, is_finish = await run_action(action_type, argument, state, docstore)

    # return observation, is_finish

    if is_finish:
        is_correct = await check_func(state.question, observation, state.key, check_llm)

        observation = f"Answer is {'CORRECT' if is_correct else 'INCORRECT'}."
        state.is_correct = is_correct
        state.finished = True

    state.scratchpad += observation or ""
    print(f"[yellow]📝 观察结果: {observation}[/yellow]")
    return observation, is_finish

async def check_answer(question: str, answer: str, key: str, llm: Callable[[str], Awaitable[str]]) -> bool:

    judge_prompt = f"""对于给定问题，判断给定的回答是否与标准答案相同。一些问题的答案取决于具体的上下文，但是你并不了解上下文，因此你应该仅仅依据给定的标准答案来判断。你应该返回True或False。\n问题： {question}\n回答： {answer}\n标准答案： {key}"""
    # print(f"[blue]📝 Judge 输入: [italic]{judge_prompt}[/italic][/blue]")
    judge_result = await llm(judge_prompt)
    print(f"[green]📝 Judge 输出: {judge_result}[/green]")
    return "true" in judge_result.lower() # type: ignore


async def run_action(action_type: str, argument: str, state: ReactAgentState, docstore: DocstoreExplorer) -> tuple[str, bool]:
    """
    运行指定的action并返回结果。

    参数:
        action_type (str): 动作的类型，可以是"Search"、"Lookup"或"Finish"。
        argument (str): 动作的参数。
        state (ReactAgentState): 当前的代理状态。

    返回:
        tuple[str, bool]: 第一个元素是运行action的结果，第二个元素表示是否得到finish结果。
    """
    if action_type == "Search":
        try:
            content = docstore.search(argument)
            state.previous_search_doc = content
            return content, False
        except Exception as e:
            return f"<CANNOT FIND THAT PAGE>", False

    elif action_type == "Lookup":
        # 🔍 检查是否有上一次搜索的文档
        if state.previous_search_doc is None:
            return "<SHOULD SEARCH FIRST>", False

        # 📝 从动作参数中获取搜索关键词
        search_term = argument.strip()
        if not search_term:
            return "<NEED PROVIDE SEARCH KEYWORD>", False

        # 🎯 在文档中执行内容搜索
        try:
            relevant_content: str = docstore.lookup(search_term)
            if relevant_content:
                return relevant_content, False
            return f"<NO RELEVANT CONTENT>", False
        except Exception as e:
            return f"<SHOULD SEARCH FIRST>", False
    elif action_type == "Finish":
        state.answer = argument
        return argument, True

    return "<INVALID ACTION> Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].", False


def read_doc(doc: str, page: int = 0, len_per_page = 400):
    # 这个函数的作用是从文档中读取指定页的内容，每页的长度由 len_per_page 参数决定。
    # 如果页码超出范围，返回空字符串。
    start_index = page * len_per_page
    end_index = start_index + len_per_page
    return doc[start_index:end_index] if start_index < len(doc) else "<END OF DOC>"


def format_agent(react_examples: str, scratchpad: str, question: str) -> str:
    prompt = f"""Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{react_examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""
    return prompt



def parse_action(string :str):
    if not string.endswith("]"):
        string += "]"
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument

    else:
        raise ValueError(f"Invalid action: {string}")
        raise ValueError(f"Invalid action: {string}")

def search_in_document(
    document: str,
    search_term: str,
    context_length: int = 400,
    score_threshold: int = 60
) -> str:
    """
    🔍 在文档中进行模糊搜索并返回相关上下文

    Args:
        document (str): 要搜索的文档内容
        search_term (str): 搜索关键词
        context_length (int): 返回内容的最大长度
        score_threshold (int): 模糊匹配的最低分数阈值(0-100)

    Returns:
        str: 找到的相关内容片段
    """
    # 🎯 核心逻辑：分段并搜索
    sentences = re.split(r'(?<=[.!?])\s+', document)
    matches: List[Tuple[int, str, float]] = []

    # 🔍 对每个句子进行模糊匹配
    for idx, sentence in enumerate(sentences):
        score = fuzz.partial_ratio(search_term.lower(), sentence.lower())
        if score >= score_threshold:
            matches.append((idx, sentence, score))

    if not matches:
        return f"<NO RELEVANT CONTENT>"

    # 📝 按相关度排序
    matches.sort(key=lambda x: x[2], reverse=True)
    best_match_idx = matches[0][0]

    # ✂️ 获取上下文窗口
    start_idx = max(0, best_match_idx - 2)  # 往前取2句
    end_idx = min(len(sentences), best_match_idx + 3)  # 往后取2句

    context = ' '.join(sentences[start_idx:end_idx])

    # 📏 如果内容过长，进行裁剪
    if len(context) > context_length:
        # 找到最后一个完整句子的位置
        last_sentence_end = max(
            context.rfind('.', 0, context_length),
            context.rfind('!', 0, context_length),
            context.rfind('?', 0, context_length)
        )
        if last_sentence_end != -1:
            context = context[:last_sentence_end + 1]
        else:
            # 如果找不到句子结束，直接截断并添加省略号
            context = context[:context_length] + '...'

    return context
