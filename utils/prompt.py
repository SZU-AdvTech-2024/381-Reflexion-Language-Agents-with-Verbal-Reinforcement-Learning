cot_reflect_agent_prompt = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Examples:
{examples}
(End of examples)

{reflections}

Relevant Context:
{context}

Question: {question}
{scratchpad}"""


cot_reflect_instruction = """你是一个高级推理智能体，能够通过自我反思进行优化改进。提供给你一个包含了相关上下文和问题的先前推理试验。这个试验中，你可能使用了Finish[<answer>]，但可能猜错了答案，或是与正确答案存在关键性表述偏差。你需要：
1. 简单诊断你可能失败或表述错误的原因
2. 制定一个新的、简洁、更高层次的计划
来缓解这类错误。

下面是一些例子:
{examples}
(End of examples)

先前的试验:
{reflections}

相关上下文:
{context}

问题:
{question}
{scratchpad}

反思："""



LAST_TRAIL_HEADER = "You previously attempted to answer the following question but failed. Below is the your last attempt."
REFLECTION_AFTER_LAST_TRIAL_HEADER = "Below is your reflection. Use it to guide you in correctly answering the given question."

COT = """Relevant Context: The Nile River is the longest river in the world, spanning approximately 6,650 kilometers (4,132 miles) in length. It flows through eleven countries in northeastern Africa, including Egypt, Sudan, and Uganda.
Question: What is the longest river in the world?
Thought: The question asks for the longest river in the world, which I know is the Nile River based on the context provided.
Action: Finish[Nile River]

Relevant Context: Ludwig van Beethoven was a German composer and pianist who was a crucial figure in the transition between the Classical and Romantic eras in Western classical music. One of his most famous compositions is the Symphony No. 9, also known as the "Choral" symphony.
Question: Which composer created Symphony No. 9?
Thought: The question is asking for the composer of Symphony No. 9. Based on the context, I know that Ludwig van Beethoven composed this symphony.
Action: Finish[Ludwig van Beethoven]

Relevant Context: Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy. During this process, plants absorb sunlight, carbon dioxide, and water to produce glucose and oxygen.
Question: What do plants produce during photosynthesis?
Thought: The question is asking about the products of photosynthesis. From the context, I know that plants produce glucose and oxygen during this process.
Action: Finish[Glucose and Oxygen]
"""

COT_REFLECT = """
Relevant Context: Ernest Hemingway's novel "The Old Man and the Sea" tells the story of Santiago, an aging Cuban fisherman, who struggles to catch a giant marlin in the Gulf Stream. The book won the Pulitzer Prize for Fiction in 1953 and contributed to Hemingway's Nobel Prize for Literature in 1954.
Question: Which literary award did "The Old Man and the Sea" contribute to Hemingway winning?
Thought: The question is asking which award "The Old Man and the Sea" contributed to Hemingway winning. Based on the context, I know the novel won the Pulitzer Prize for Fiction and contributed to his Nobel Prize for Literature.
Action: Finish[Pulitzer Prize for Fiction]

Reflection: My answer was correct based on the context, but may not be the exact answer stored by the grading environment. Next time, I should try to provide a less verbose answer like "Pulitzer Prize" or "Nobel Prize."

Context: On 14 October 1947, Chuck Yeager, a United States Air Force test pilot, became the first person to break the sound barrier by flying the Bell X-1 experimental aircraft at an altitude of 45,000 feet.
Charles Elwood "Chuck" Yeager (13 February 1923 - 7 December 2020) was a United States Air Force officer, flying ace, and test pilot. He is best known for becoming the first person to break the sound barrier, which he achieved in the Bell X-1 aircraft named Glamorous Glennis. Yeager was also a distinguished fighter pilot during World War II and was credited with shooting down at least 12 enemy aircraft. In 1973, he was inducted into the National Aviation Hall of Fame for his significant contributions to aviation.
Question: Who is the first person to break the sound barrier?
Thought: The question is asking for the first person to break the sound barrier. From the context, I know that Chuck Yeager, a United States Air Force test pilot, was the first person to break the sound barrier.
Action: Finish[Chuck Yeager]

Reflection: Upon reflecting on the incorrect answer I provided, I realize that I may not have given the full name of the individual in question. In the context, both the given name and the nickname were mentioned, and I only used the nickname in my response. This could have been the reason my answer was deemed incorrect. Moving forward, when attempting this question again or similar questions, I will make sure to include the complete name of the person, which consists of their given name, any middle names, and their nickname (if applicable). This will help ensure that my answer is more accurate and comprehensive."""


LAST_ATTEMPT_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'


REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""


REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'

REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'


REACT_REFLECT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""
