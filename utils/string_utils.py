import re
from utils.prompt import LAST_TRAIL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER

def format_step(step: str):
    """
    格式化步骤字符串

    参数:
        step: 输入的步骤字符串

    返回:
        格式化后的字符串,会移除首尾的换行符和空格,并将中间的换行符替换为空字符串
    """
    return step.strip("\n").strip().replace("\n", "")

def parse_action(action: str):
    """
    解析动作字符串
    """
    # pattern = r'^(\w+)\[(.+)\]$'
    pattern = r'^(\w+)\[(.+)\]'
    match = re.match(pattern, action)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument

    else:
        return None, None

def format_last_attempt(question: str,
                        scratchpad: str,
                        header = LAST_TRAIL_HEADER):
    """
    格式化最后一次尝试
    """
    formatted_scratchpad = truncate_scratchpad(scratchpad).strip('\n').strip()
    return f"{header}\nQuestion: {question}\n{formatted_scratchpad}\n(END PREVIOUS TRIAL)\n"

def truncate_scratchpad(scratchpad: str, max_length: int = 2000) -> str:
    """
    截断scratchpad以确保其长度不超过限制。

    具体做法:
    1. 将scratchpad按行分割
    2. 找出所有以'Observation'开头的行
    3. 按照每行字符长度对这些observation行进行排序
    4. 如果总长度超过限制,就从最长的observation开始截断
    5. 截断方式是将observation内容替换为[truncated]
    6. 重复直到总长度在限制内
    """
    lines = scratchpad.split('\n')
    observations = [line for line in lines if line.startswith('Observation')]
    observations_by_length = sorted(observations, key=len)

    while len('\n'.join(lines)) > max_length and observations_by_length:
        largest_observation = observations_by_length.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated]'

    return '\n'.join(lines)


def format_reflections(reflections: list[str], header = REFLECTION_AFTER_LAST_TRIAL_HEADER) -> str:
    if reflections == []:
        return ""
    else:
        return header + 'Reflections:\n- '+ '\n- '.join([r.strip() for r in reflections])