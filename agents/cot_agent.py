from enum import Enum
from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import PromptTemplate
from utils.prompt import cot_reflect_agent_prompt, cot_reflect_instruction, COT, COT_REFLECT
from utils.llms import local_llm
from utils.string_utils import format_step, parse_action, format_last_attempt, format_reflections
from rich import print, box
from rich.console import Console
from rich.table import Table
from pydantic import BaseModel
from typing import List, Tuple, Callable, Awaitable


class CoTAgentStrategy(Enum):
    COT_ONLY = "COT_ONLY"   # 仅使用COT，通过thought，回答 # 只有一次机会
    COT_GT = "COT_GT"       # 添加可靠的相关信息来回答      # 只有一次机会
    COT_REFLEXION = "COT_REFLEXION"  # 无context，但有反思机制
    COT_GT_EPM = "COT_GT_EPM"    # 添加可靠的相关信息，如果回答错误，可以参考上次的错误进行重试
    COT_GT_REFLEXION = "COT_GT_REFLEXION"   # 添加可靠的相关信息，如果回答错误，进行反思，重试
    COT_GT_EPM_REFLEXION = "COT_GT_EPM_REFLEXION"   # 添加可靠的相关信息，如果回答错误，进行反思，并可以参考上次的错误进行重试

class CotAgentState(BaseModel):
    question: str
    context: str | None
    key: str
    answer: str = ""
    step_n: int = 0
    finished: bool = False
    scratchpad: str = ""
    reflections: List[str] = []
    reflections_str: str = ""
    previous_attempts: List[str] = []  # 记录之前的尝试
    error_summary: str = ""           # 错误总结
    max_step: int = 10
    is_correct: bool | None = None
    strategy: CoTAgentStrategy = CoTAgentStrategy.COT_ONLY

async def run_cot_agent(
    question: str,
    key: str,
    strategy: CoTAgentStrategy,
    context: str | None,
    action_llm: Callable[[str], Awaitable[str]],
    reflect_llm: Callable[[str], Awaitable[str]],
    judge_llm: Callable[[str], Awaitable[str]],
    max_step: int = 10,
) -> CotAgentState:
    print(f"🚀 开始运行 CoT Agent - 策略: {strategy.value}")
    print(f"❓ 问题: {question}")
    print(f"🔑 答案: {key}")

    state = CotAgentState(
        question=question,
        context=context,
        key=key,
        max_step=max_step,
        strategy=strategy
    )

    if strategy == CoTAgentStrategy.COT_ONLY or strategy == CoTAgentStrategy.COT_GT:
        print("📝 单次推理模式")
        state = await step_cot_agent(state, action_llm, reflect_llm, judge_llm)
        return state

    print("🔄 多轮推理模式")
    while not state.finished and state.step_n < max_step:
        state = await step_cot_agent(state, action_llm, reflect_llm, judge_llm)
        print("检查状态", state.is_correct)

        # 如果答案正确，直接结束
        if state.is_correct:
            state.finished = True
            break

        # EPM 策略的特殊处理
        if not state.is_correct and state.answer:
            if strategy in [CoTAgentStrategy.COT_GT_EPM, CoTAgentStrategy.COT_GT_EPM_REFLEXION]:
                print("🔄 错误记忆模式：重置状态并保留错误记忆")
                # 重置状态，保留错误记忆
                state.scratchpad = ""
                state.finished = False
                state.answer = ""
                state.is_correct = None
            elif strategy == CoTAgentStrategy.COT_REFLEXION:
                print("🔄 纯反思模式：重置状态")
                # 重置状态
                state.scratchpad = ""
                state.finished = False
                state.answer = ""
                state.is_correct = None

    return state

async def step_cot_agent(
    state: CotAgentState,
    action_llm: Callable[[str], Awaitable[str]],
    reflect_llm: Callable[[str], Awaitable[str]],
    judge_llm: Callable[[str], Awaitable[str]],
) -> CotAgentState:
    new_state = state.model_copy()
    new_state.step_n += 1

    print("🤔 思考中...")
    thought = await think(new_state, action_llm)

    print("🎯 执行动作...")
    action = await act(new_state, action_llm)

    print("👀 观察结果...")
    observation = await observe(new_state, action, judge_llm)

    # 处理错误情况
    if not new_state.is_correct and new_state.answer:
        if state.strategy in [CoTAgentStrategy.COT_GT_EPM]:
            # EPM 策略：只记录错误，不反思
            print("🔄 错误记忆模式...")
            new_state = await reflect(new_state, reflect_llm)
        elif state.strategy in [CoTAgentStrategy.COT_REFLEXION, CoTAgentStrategy.COT_GT_REFLEXION, CoTAgentStrategy.COT_GT_EPM_REFLEXION]:
            # Reflection 策略：进行反思
            print("🔄 开始反思...")
            new_state = await reflect(new_state, reflect_llm)

    # 如果答案正确或达到最大步数，标记为完成
    if new_state.is_correct or new_state.step_n >= new_state.max_step:
        print("✅ 完成" if new_state.is_correct else "⚠️ 达到最大步数限制")
        new_state.finished = True

    return new_state

async def think(
    state: CotAgentState,
    llm: Callable[[str], Awaitable[str]]
) -> str:
    state.scratchpad += "\nThought:"
    prompt = build_agent_prompt(state)
    thought = await llm(prompt)
    state.scratchpad += " " + format_step(thought)
    print(f"💭 思考结果: {thought}")
    return thought

async def act(
    state: CotAgentState,
    llm: Callable[[str], Awaitable[str]]
) -> str:
    state.scratchpad += "\nAction:"
    prompt = build_agent_prompt(state)
    action = await llm(prompt)
    state.scratchpad += " " + format_step(action)
    print(f"🎯 执行动作: {action}")
    return action

async def observe(
    state: CotAgentState,
    action: str,
    judge_llm: Callable[[str], Awaitable[str]]
) -> str:
    state.scratchpad += "\nObservation:"
    action_type, argument = parse_action(action)

    if action_type == "Finish":
        state.answer = argument or ""
        state.is_correct = await check_answer(state.question, state.answer, state.key, judge_llm)
        observation = "Answer is " + ("CORRECT" if state.is_correct else "INCORRECT")
        state.scratchpad += " " + observation
        print(f"📝 回答: {state.answer}")
        print(f"✅ 正确性: {observation}")
        return observation

    print("❌ 无效动作")
    return "<INVALID ACTION>"

async def reflect(
    state: CotAgentState,
    reflect_llm: Callable[[str], Awaitable[str]]
) -> CotAgentState:
    # EPM 策略：记录错误尝试和生成错误总结
    if state.strategy in [CoTAgentStrategy.COT_GT_EPM, CoTAgentStrategy.COT_GT_EPM_REFLEXION]:
        # 记录错误尝试
        state.previous_attempts.append(state.scratchpad)
        # 生成错误总结
        error_summary_prompt = f"""分析以下解题尝试中的错误模式：
        问题：{state.question}
        尝试：{state.scratchpad}
        标准答案：{state.key}
        请总结错误的关键点。"""
        state.error_summary = await reflect_llm(error_summary_prompt)
        print(f"🔍 错误总结: {state.error_summary}")

        # 如果是纯 EPM 策略，将错误总结添加到 reflections_str
        if state.strategy == CoTAgentStrategy.COT_GT_EPM:
            state.reflections_str = f"\n之前尝试中的错误总结：{state.error_summary}"

    # Reflection 策略：进行反思
    if state.strategy in [CoTAgentStrategy.COT_REFLEXION, CoTAgentStrategy.COT_GT_REFLEXION, CoTAgentStrategy.COT_GT_EPM_REFLEXION]:
        # 反思逻辑
        state.reflections_str = format_last_attempt(state.question, state.scratchpad)
        prompt = build_reflect_prompt(state)
        reflection = await reflect_llm(prompt)
        state.reflections = [format_step(reflection)]
        state.reflections_str = "\n" + format_reflections(state.reflections)
        print(f"🤔 反思结果: {reflection}")

    return state


async def check_answer(
    question: str,
    answer: str,
    key: str,
    llm: Callable[[str], Awaitable[str]]
) -> bool:
    print("🔍 检查答案...")
    judge_prompt = f"""对于给定问题，判断给定的回答是否与标准答案相同。一些问题的答案取决于具体的上下文，但是你并不了解上下文，因此你应该仅仅依据给定的标准答案来判断。你应该返回True或False。\n问题： {question}\n回答： {answer}\n标准答案： {key}"""
    judge_result = await llm(judge_prompt)
    result = "true" in judge_result.lower()
    print(f"✨ 判断结果: {result}")
    return result

def build_agent_prompt(state: CotAgentState) -> str:
    # 确定是否使用 context
    use_context = state.strategy not in [CoTAgentStrategy.COT_ONLY, CoTAgentStrategy.COT_REFLEXION]
    context = state.context if use_context else "<EMPTY>"

    # 对于 EPM 策略，reflections_str 已经包含了错误总结，不需要额外添加
    return cot_reflect_agent_prompt.format(
        examples=COT,
        context=context,
        reflections=state.reflections_str,
        question=state.question,
        scratchpad=state.scratchpad
    )

def build_reflect_prompt(state: CotAgentState) -> str:
    use_context = state.strategy not in [CoTAgentStrategy.COT_ONLY, CoTAgentStrategy.COT_REFLEXION]
    return cot_reflect_instruction.format(
        examples=COT_REFLECT,
        context=state.context if use_context else "<EMPTY>",
        question=state.question,
        scratchpad=state.scratchpad,
        reflections=state.reflections_str
    )