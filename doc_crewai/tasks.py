# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         tasks
# Description:  任务
# Author:       shaver
# Date:         2025/5/20
# -------------------------------------------------------------------------------
from textwrap import dedent

from crewai import Task



class APITasks:
    def requirements_task(self, agent):
        task = Task(
        description="调用工具获取需求信息，分解出详细的测试需求列表",
        expected_output="详细的需求列表",
        agent=agent,
        )
        return task

    def testcase_task(self, agent):
        task = Task(
            description="根据需求列表，编写测试用例，并进行测试",
            expected_output="详细的测试用例列表",
            agent=agent,
        )
        return task

    def code_task(self, agent):
        task = Task(
            description="根据测试用例，编写测试代码，尽量使用pytest",
            expected_output="完整的pytest脚本",
            agent=agent,
        )
        return task

    def review_task(self, agent):
        task = Task(
            description="使用python+pytest框架知识检查脚本",
            expected_output="最终答案是完整的pytest脚本",
            agent=agent,
        )
        return task

    def file_generate_task(self, agent):
        task = Task(
            description="调用Python file Generator工具生成Python代码保存到文件中",
            expected_output="最终答案是完整的pytest脚本",
            agent=agent,
        )
        return task