from enum import Enum
from typing import Awaitable, Callable
import uuid
from agents.react_agent import ReactAgentState, act, check_answer, observe, think
from langchain.chat_models.base import BaseChatModel

from utils.fewshots import REFLECTIONS, WEBTHINK_SIMPLE3
from utils.prompt import LAST_ATTEMPT_HEADER, REACT_REFLECT_INSTRUCTION, REFLECT_INSTRUCTION, REFLECTION_AFTER_LAST_TRIAL_HEADER, REFLECTION_HEADER
from pydantic import BaseModel
from langchain.agents.react.base import DocstoreExplorer
from agents.action_runner import create_wikipedia_docstore
from rich import print

class ReflectionType(Enum):
    NONE = "base"
    LAST_ATTEMPT = "last_attempt"
    REFLEXION = "reflexion"
    LAST_ATTEMPT_AND_REFLEXION = "last_attempt_and_reflexion"


class ReactReflectAgentState(ReactAgentState):
    reflections: list[str] = [] # 反思记录
    reflections_str: str = "" # 反思记录字符串
    trials_count: int = 0 # 当前尝试次数


class ReactReflectRecord(BaseModel):
    id: str = str(uuid.uuid4())
    question: str           # 从原始输入中复制过来
    key: str                # 从原始输入中复制过来
    answers: list[str]      # 记录每一轮llm所提供的finish回答（如果这一轮没有提供，则记录空字符串）
    is_correct: bool | None # 记录最终答案是否正确（有可能没有触发任何一次Finish，就是None）
    reflections: list[str]  # 记录每一次反思，如果这一轮没有反思，则记录空字符串
    step_n: int = 0         # 记录总共运行了多少步
    trials_count: int = 0   # 记录总共尝试了多少次
    # searchs: list[str]     # 记录每一次搜索的参数
    # searchs_results: str   # 记录每一次搜索的结果


async def run_react_reflect_agent(
    question: str,
    key: str,
    llm: Callable[[str], Awaitable[str]],
    check_llm: Callable[[str], Awaitable[str]] | None = None,
    strategy: ReflectionType = ReflectionType.NONE,
    max_steps: int = 6,  # 每轮最多执行步数,超过触发反思    # 这里的step是指每轮执行的最多步数
    trials_n: int = 5,   # 最大尝试次数,包含反思        # 这里的trials是指反思的次数
    id: str | None = None,
) -> ReactReflectRecord:
    # 🏃‍♂️ 初始化状态和记录
    state = ReactReflectAgentState(question=question, key=key)
    record = ReactReflectRecord(
        question=question,
        key=key,
        answers=[],
        is_correct=None,
        reflections=[],
        step_n=0,
        trials_count=0
    )
    if id is not None:
        record.id = id

    docstore = create_wikipedia_docstore()

    # 🔄 主循环 - 最多尝试trials_n次
    while state.trials_count < trials_n:
        try:
            # 📝 每轮开始前重置状态
            if state.error:
                state.scratchpad += "\n" + state.error + "\n"
                state.error = None
                state.step_n = 0


            # 如果不是第一次尝试，则需要对之前的步骤进行反思
            if state.trials_count > 0:
                await reflect(state, llm, strategy)
                state.scratchpad = ""

            # 🎯 执行当前轮次
            while True:
                # not state.finished and state.step_n < max_steps:
                state = await step_react_reflect_agent(
                    state,
                    llm,
                    docstore,
                    check_llm,
                    strategy,
                )

                # 📝 更新记录
                if state.answer:
                    record.answers.append(state.answer)
                    state.answer = ""
                record.step_n = state.step_n
                record.is_correct = state.is_correct
                record.trials_count = state.trials_count

                if state.finished or state.step_n >= max_steps:
                    break

            # ✅ 如果答案正确或达到最大尝试次数,结束循环
            if state.is_correct or state.trials_count >= trials_n:
                break

            # 🔄 否则进入下一轮尝试
            state.trials_count += 1
            state.step_n = 0
            state.finished = False

        except Exception as e:
            # print(f"[red]❌ 步骤执行出错: {str(e)}[/red]")
            state.error = "<ERROR, PLEASE OUTPUT ACCORDING TO THE EXAMPLES>"
            continue

    # 🎯 完成运行
    state.finished = True
    # print("[green]🎉 结束[/green]")
    # print(f"[blue]📝 运行了 {state.step_n} 步, {state.trials_count} 轮[/blue]")

    return record


async def step_react_reflect_agent(
    state: ReactReflectAgentState,
    llm: Callable[[str], Awaitable[str]],
    docstore: DocstoreExplorer,
    check_llm: Callable[[str], Awaitable[str]] | None = None,
    reflection_type: ReflectionType = ReflectionType.NONE,
    agent_format_func: Callable[[ReactReflectAgentState], str] = lambda x: format_agent(WEBTHINK_SIMPLE3, x.scratchpad, x.question, x.reflections_str),
) -> ReactReflectAgentState:

    new_state = state.model_copy()
    new_state.step_n += 1

    print(f"[blue]📝 进入第 {new_state.step_n} 步[/blue]")
    # 🤖 执行核心步骤
    await think(new_state, llm, agent_format_func) # type: ignore
    action = await act(new_state, llm, agent_format_func) # type: ignore
    observation, is_finish = await observe(new_state, action, llm, check_func=check_answer, check_llm=check_llm or llm, docstore=docstore)

    if is_finish and not new_state.is_correct and reflection_type != ReflectionType.NONE:
        # 🤔 错误答案触发反思
        await reflect(new_state, llm, reflection_type)
        new_state.finished = False

    return new_state


async def reflect(state: ReactReflectAgentState, llm: Callable[[str], Awaitable[str]], strategy: ReflectionType) -> None:
    # print(f"[blue]📝 正在反思...[/blue]")
    if strategy == ReflectionType.LAST_ATTEMPT:
        state.reflections = [state.scratchpad]
        state.reflections_str = format_last_attempt(state.question, state.scratchpad)

    elif strategy == ReflectionType.REFLEXION:
        prompt = build_reflextion_prompt(state.question, state.scratchpad)
        reflection = await llm(prompt +"\n(Note: Write down your reflection in one line without Reflection prefix.)")
        state.reflections = [reflection]
        state.reflections_str = format_reflection(state.reflections)

    elif strategy == ReflectionType.LAST_ATTEMPT_AND_REFLEXION:
        state.reflections_str = format_last_attempt(state.question, state.scratchpad)
        prompt = build_reflextion_prompt(state.question, state.scratchpad)
        reflection = await llm(prompt +"\n(Note: Write down your reflection in one line without Reflection prefix.)")
        state.reflections = [reflection]
        state.reflections_str += "\n" + format_reflection(state.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
    else:
        raise ValueError(f"Invalid reflection strategy: {strategy}")

    # print(f"[blue]📝 反思: {state.reflections_str}[/blue]")

def build_reflextion_prompt(question: str, scratchpad: str) -> str:
    return REFLECT_INSTRUCTION.format(question=question, scratchpad=scratchpad, examples=REFLECTIONS)


def format_last_attempt(question: str, scratchpad: str) -> str:
    return LAST_ATTEMPT_HEADER + f"Question: {question}\n" + f"{scratchpad}\n" + f"\n<END PREVIOUS ATTEMPT>"

def format_reflection(reflections: list[str], header = REFLECTION_HEADER) -> str:
    if len(reflections) == 0:
        return ""
    else:
        return header + "Reflections:\n- " + "\n-".join(r.strip() for r in reflections)

def format_step(step: str):
    return step.strip('\n').strip().replace('\n', '')


def format_agent(react_examples: str, scraptchpad: str, question: str, reflections: str)-> str:
    return REACT_REFLECT_INSTRUCTION.format(examples=react_examples, reflections=reflections, question=question, scratchpad=scraptchpad)