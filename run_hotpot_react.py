import joblib
import pandas as pd
import numpy as np

import json
from pathlib import Path
import asyncio


from agents.react_reflect_agent import ReflectionType, run_react_reflect_agent
from utils.llms import create_llm_invoker, local_llm, openai_llm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

max_steps = 7
trials_n = 5
strategy = ReflectionType.LAST_ATTEMPT_AND_REFLEXION

log_file = f"output/hotpot_react_reflexion_{strategy.value}_4o_mini_nostop.log"
records_file = f"output/hotpot_react_reflexion_{strategy.value}_4o_mini_nostop.json"

# alocal_llm = create_llm_invoker(local_llm, stop=["\n"])
# aopenai_llm = create_llm_invoker(openai_llm, stop=["\n"])
alocal_llm = create_llm_invoker(local_llm)
aopenai_llm = create_llm_invoker(openai_llm)


inference_llm = aopenai_llm
check_llm = alocal_llm






hotpot_sample_file = "data/hotpot-qa-distractor-sample.joblib"
hotpot: pd.DataFrame = joblib.load(hotpot_sample_file).reset_index(drop=True)
print("len(hotpot):", len(hotpot))
hotpot['supporting_paragraphs'] = None
for ind, row in hotpot.iterrows():
    # 获取支持性文章标题和上下文信息
    supporting_articles = row['supporting_facts']['title']  # 支持性文章标题列表
    articles = row['context']['title']                     # 所有文章标题
    sentences = row['context']['sentences']                # 所有文章句子

    # 🎯 提取支持段落
    # 对每个支持性文章,找到对应的句子并拼接成段落
    supporting_paragraphs = []
    for article in supporting_articles:
        # 使用numpy where找到文章对应的句子位置
        # 拼接该文章的所有句子形成段落
        supporting_paragraph = ''.join(sentences[np.where(articles == article)][0])
        supporting_paragraphs.append(supporting_paragraph)

    # 用换行符连接多个支持段落
    supporting_paragraphs = '\n\n'.join(supporting_paragraphs)
    hotpot.at[ind, 'supporting_paragraphs'] = supporting_paragraphs




# 🎯 使用 tenacity 装饰器进行重试
@retry(
    # 最多重试3次
    stop=stop_after_attempt(max_attempt_number=3),
    # 指数退避策略,初始等待1秒,最长等待10秒
    wait=wait_exponential(multiplier=1, min=1, max=10),
    # 只对特定异常进行重试
    # retry=retry_if_exception_type(Exception),
    # 重试时打印错误信息
    before_sleep=lambda retry_state: print(f"[red]❌ 第{retry_state.attempt_number}次尝试失败,等待重试...[/red]"),
    # 重试全部失败后返回None和error信息
    retry_error_callback=lambda retry_state: (None, str(retry_state.outcome))
)
async def run_row(row: pd.Series, ind: int):
    # try:
    print("--------------------------------")
    print(f"🧠 问题 {ind+1} : {row['question']}") # type: ignore
    question = row['question']
    key = row['answer']
    record = await run_react_reflect_agent(
        id=row['id'],
        question=question,
        key=key,
        llm=inference_llm,
        check_llm=check_llm,
        strategy=strategy,
        max_steps=max_steps,
        trials_n=trials_n
    )

    log_info = ""
    log_info += f"🧠 问题 {ind+1} : {question}\n"
    log_info += f"🧠 问题 {ind+1} 的答案: {key}\n"
    log_info += f"🧠 问题 {ind+1} 的回答: {record.answers}\n"
    log_info += f"🧠 回答是否正确: {record.is_correct}\n"
    log_info += f"🧠 步数: {record.step_n}\n"
    log_info += f"🧠 反思: {record.reflections}\n"
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
            records_json = [r.model_dump() if r is not None else {"error": "处理失败"}
                          for r in answer_records]
            with open(records_file, "w", encoding="utf-8") as f:
                json.dump(records_json, f, ensure_ascii=False, indent=2)

            print(f"[green]✅ 工作者{worker_id}完成第{ind+1}条数据[/green]")

            # 标记任务完成
            queue.task_done()

        except asyncio.CancelledError:
            break

async def run_all():
    """
    🎯 主控制流程
    """
    output_dir = Path("output")

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
