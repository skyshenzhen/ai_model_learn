# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         main
# Description:
# Author:       shaver
# Date:         2025/5/20
# -------------------------------------------------------------------------------

from textwrap import dedent
from crewai import Agent, Task, Crew, LLM

llm = LLM(
    model="deepseek-chat",
    temperature=0.7,
    base_url="https://api.deepseek.com/v1",
    api_key="sk-aa3a24ed00744632943daf833f298803"
)

#---------------------------------智能体---------------------------------
# 软件开发工程师
software_development_engineer = Agent(
    role="高级软件开发工程师",
    goal="按需创建软件",
    backstory=dedent("""
        你是一家领先的技术智库的高级软件工程师。
        你在Python编程方面有专长，并尽你所能长产出完美的代码"""),
    allow_delegation=False, # 不允许委托
    verbose=True,
    llm=llm
)

# 软件测试工程师
software_testing_engineer = Agent(
    role="高级软件测试工程师",
    goal="提升软件质量",
    backstory=dedent("""
        你是一位高级软件测试工程师，负责测试软件的质量。
        你擅长找到Python代码中的bug"""),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# 首席软件质量控制工程师
chief_software_quality_control_engineer = Agent(
    role="首席软件质量控制工程师",
    goal="确保软件质量",
    backstory=dedent("""
        你是一位资深的软件质量控制工程师，负责确保软件的质量。
        你擅长使用Python进行自动化测试"""),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# 运行智能体
run_person=Agent(
    role="运行智能体",
    goal="运行游戏智能体",
    backstory=dedent("""
        你是一位资深的游戏智能体开发者，负责运行游戏智能体。
        你讲上面生成的游戏代码，保存到一个文件中，并运行游戏。"""),
    allow_delegation=False,
    verbose=True,
    llm=llm
)


#---------------------------------确定要创建的游戏及相关指令---------------------------------
print("欢迎来到游戏智能体开发平台")
print("-----------------------------")
game=input("你想构建什么样的游戏？游戏的基本要求有哪些？\n")

#---------------------------------创建任务---------------------------------
# 任务1：编写游戏的基本框架
code_task=Task(
    description=dedent(f"""您将使用python编程语言来创建一个游戏，以下是指令：
     指令
     --------------
     {game}
     """),
    expected_output="您的最终答案必须是完整的python代码，仅包含python代码，别无其他",
    agent=software_development_engineer,
)

# 任务2：测试游戏的基本框架
test_task=Task(
    description=dedent("""您将测试游戏的基本框架，以下是指令：
     指令
     --------------
     请测试游戏的基本框架是否正确运行，并给出测试结果。
     """),
    expected_output="测试结果必须包含测试游戏的基本框架是否正确运行的结果，以及测试结果的评价",
    agent=software_testing_engineer,
)

# 任务3：游戏质量的控制
quality_control_task=Task(
    description=dedent("""您将对游戏的质量进行控制，以下是指令：
     指令
     --------------   
     请对游戏的质量进行控制，确保游戏的质量达到最高标准。
     """),
    expected_output="测试结果必须包含对游戏的质量进行控制的结果，以及控制结果的评价",
    agent=chief_software_quality_control_engineer,
)

# 任务4：保存文件 运行游戏
run_task=Task(
    description=dedent("""您将保存文件并运行游戏，以下是指令：
     指令
     -   
     请保存文件并运行游戏。
     """),
    expected_output="需要保证游戏正常运行",
    agent=run_person,
    is_final=True
)

#---------------------------------智能体编排---------------------------------
crew= Crew(
    agents=[software_development_engineer,
            software_testing_engineer,
            chief_software_quality_control_engineer,
            run_person
            ],
    tasks=[code_task,
           test_task,
           quality_control_task,
           run_task
           ],
    verbose=True
)

#---------------------------------启动智能体运行----------------
game=crew.kickoff()

# 打印结果
print("游戏智能体开发平台-开始运行")
print(game)

print("游戏智能体开发平台-结束运行")