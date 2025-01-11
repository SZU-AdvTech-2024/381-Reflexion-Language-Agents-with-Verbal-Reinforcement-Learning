import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
import asyncio

from agents.cot_agent import CoTAgentStrategy, run_cot_agent
from utils.llms import create_llm_invoker, local_llm, openai_llm
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置参数
max_steps = 5
strategy = CoTAgentStrategy.COT_GT_EPM

log_file = f"output/hotpot_cot_{strategy.value}_4o_mini.log"
records_file = f"output/hotpot_cot_{strategy.value}_4o_mini.json"

# 创建 LLM 调用器
alocal_llm = create_llm_invoker(local_llm)
aopenai_llm = create_llm_invoker(openai_llm)

inference_llm = aopenai_llm
check_llm = alocal_llm

# 加载数据
hotpot_sample_file = "data/hotpot-qa-distractor-sample.joblib"
hotpot: pd.DataFrame = joblib.load(hotpot_sample_file).reset_index(drop=True)
print("len(hotpot):", len(hotpot))

# 处理支持性段落
hotpot['supporting_paragraphs'] = None
for ind, row in hotpot.iterrows():
    # 获取支持性文章标题和上下文信息
    supporting_articles = row['supporting_facts']['title']  # 支持性文章标题列表
    articles = row['context']['title']                     # 所有文章标题
    sentences = row['context']['sentences']                # 所有文章句子

    # 🎯 提取支持段落
    supporting_paragraphs = []
    for article in supporting_articles:
        # 使用numpy where找到文章对应的句子位置
        supporting_paragraph = ''.join(sentences[np.where(articles == article)][0])
        supporting_paragraphs.append(supporting_paragraph)

    # 用换行符连接多个支持段落
    supporting_paragraphs = '\n\n'.join(supporting_paragraphs)
    hotpot.at[ind, 'supporting_paragraphs'] = supporting_paragraphs

# 🎯 使用 tenacity 装饰器进行重试
@retry(
    stop=stop_after_attempt(max_attempt_number=3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=lambda retry_state: print(f"[red]❌ 第{retry_state.attempt_number}次尝试失败,等待重试...[/red]"),
    retry_error_callback=lambda retry_state: (None, str(retry_state.outcome))
)
async def run_row(row: pd.Series, ind: int):
    print("--------------------------------")
    print(f"🧠 问题 {ind+1} : {row['question']}")
    question = row['question']
    key = row['answer']
    context = row['supporting_paragraphs']

    state = await run_cot_agent(
        question=question,
        key=key,
        context=context,
        strategy=strategy,
        action_llm=inference_llm,
        reflect_llm=inference_llm,
        judge_llm=check_llm,
        max_step=max_steps,
    )

    # 构建记录
    record = {
        "id": row['id'],
        "question": question,
        "key": key,
        "answers": [state.answer] if state.answer else [],
        "is_correct": state.is_correct,
        "step_n": state.step_n,
        "reflections": state.reflections,
        "scratchpad": state.scratchpad
    }

    # 构建日志信息
    log_info = ""
    log_info += f"🧠 问题 {ind+1} : {question}\n"
    log_info += f"🧠 问题 {ind+1} 的答案: {key}\n"
    log_info += f"🧠 问题 {ind+1} 的回答: {state.answer}\n"
    log_info += f"🧠 回答是否正确: {state.is_correct}\n"
    log_info += f"🧠 步数: {state.step_n}\n"
    log_info += f"🧠 反思: {state.reflections}\n"
    log_info += "\n"

    return record, log_info

async def worker(worker_id: int,
                queue: asyncio.Queue,
                answer_records: list,
                all_logs: list,
                records_file: str):
    """
    🤖 工作者协程
    """
    while True:
        try:
            # 从队列获取任务
            ind, row = await queue.get()

            # 处理任务
            record, log_info = await run_row(row, ind)

            # 更新结果
            answer_records.append(record)
            all_logs.append(log_info)

            # 保存进度
            with open(records_file, "w", encoding="utf-8") as f:
                json.dump(answer_records, f, ensure_ascii=False, indent=2)

            print(f"[green]✅ 工作者{worker_id}完成第{ind+1}条数据[/green]")

            # 标记任务完成
            queue.task_done()

        except asyncio.CancelledError:
            break

async def run_all():
    """
    🎯 主控制流程
    """
    # 共享状态
    answer_records = []
    all_logs = []

    # 创建任务队列
    queue = asyncio.Queue()

    # 创建工作者
    workers = []
    worker_num = 10
    for i in range(worker_num):
        worker_task = asyncio.create_task(
            worker(i, queue, answer_records, all_logs, records_file)
        )
        workers.append(worker_task)

    # 添加所有任务到队列
    for ind, row in hotpot.iterrows():
        queue.put_nowait((ind, row))

    # 等待所有任务完成
    await queue.join()

    # 取消所有工作者
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    # 保存最终日志
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_logs))
    print(f"[green]✅ 已保存log到{log_file}[/green]")

if __name__ == "__main__":
    asyncio.run(run_all())