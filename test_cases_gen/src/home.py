import json
import re
import sys
import time

import streamlit as st
from io import BytesIO

from docx import Document
from generate_by_txt import GenerateByTxt
from generate_by_swagger import GenerateBySwagger


class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index
        self.task_values = []

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(
                self.colors)  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain",
                                                f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "Market Research Analyst" in cleaned_data:
            # Apply different color
            cleaned_data = cleaned_data.replace("Market Research Analyst",
                                                f":{self.colors[self.color_index]}[Market Research Analyst]")
        if "Business Development Consultant" in cleaned_data:
            cleaned_data = cleaned_data.replace("Business Development Consultant",
                                                f":{self.colors[self.color_index]}[Business Development Consultant]")
        if "Technology Expert" in cleaned_data:
            cleaned_data = cleaned_data.replace("Technology Expert",
                                                f":{self.colors[self.color_index]}[Technology Expert]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


def vision_page():
    st.title("测试用例生成器")
    # 单选框  有文本生成 和 swagger生成
    st.subheader("选择生成方式")
    generate_type = st.radio("", ["文本生成", "swagger生成"])

    bytes_data = None
    MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB
    str_desc = ''

    if generate_type == "文本生成":
        str_desc = "请输入需求文档或手动输入文本"
        st.caption("可识别TXT需求文档或是通过手动输入文本，并自动生成测试用例。")
        uploaded_file = st.file_uploader("请选择文件进行上传", type=None)
        if uploaded_file is not None:
            try:
                # 检查文件类型
                if uploaded_file.type not in ['text/plain']:
                    st.error("请上传TXT文件")
                    st.stop()

                # 检查文件大小
                if uploaded_file.size > MAX_FILE_SIZE:
                    st.error(f"上传文件过大，请上传小于 {MAX_FILE_SIZE / 1024 / 1024}MB 的文件")
                    st.stop()
                else:
                    bytes_data = uploaded_file.read()

            except Exception as e:
                st.error(f"文件处理出错: {str(e)}")
    elif generate_type == "swagger生成":
        str_desc = "请输入swagger链接"
        st.caption("可识别swagger接口文档，并自动生成测试用例。")

    # 增加文本输入框
    text_input = st.text_area(str_desc, height=200)

    if st.button("Create Test Cases"):
        # 检查文件和文本输入框是否都为空
        if bytes_data is None and text_input == "":
            st.error("Please upload an image or enter text.")
            st.stop()
        # 检查文件和文本是否都含有
        if bytes_data is not None and text_input != "":
            st.error("Please upload an image or enter text, not both.")
            st.stop()

        stopwatch_placeholder = st.empty()
        # 将bytes_data转回str
        if bytes_data is not None:
            bytes_data = bytes_data.decode('utf-8')
        else:
            bytes_data = text_input

        start_time = time.time()
        # 初始化StreamToExpander
        stream_handler = StreamToExpander(st)

        with st.expander("任务执行过程......"):
            sys.stdout = stream_handler
            with st.spinner("Generating Results"):
                if generate_type == "swagger生成":
                    crew_result = GenerateBySwagger.create_crewai_setup(bytes_data)
                else:
                    crew_result = GenerateByTxt.create_crewai_setup(bytes_data)

        end_time = time.time()
        total_time = end_time - start_time
        stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")

        st.header("任务执行情况:")
        if stream_handler.task_values:
            st.table({"任务": stream_handler.task_values})

        # 替换原来的docx生成代码为以下Excel生成代码
        if crew_result:
            st.header("Results:")
            crew_result = str(crew_result)
            # 匹配json代码块
            pattern = r'```json(.*?)```'

            match = re.search(pattern, crew_result, re.DOTALL)
            if match:
                crew_result = match.group(1).strip()
            # 转为json
            crew_result = json.loads(crew_result)

            if isinstance(crew_result, dict) and 'module' in crew_result and 'test_cases' in crew_result:
                # 生成Excel文件
                import pandas as pd
                from io import BytesIO
                from pyexcelerate import Workbook

                # 准备数据
                rows = []
                for case in crew_result['test_cases']:
                    # 步骤描述
                    steps_desc_str = "\n".join(
                        [f"{s['step']}. {s['action']}\n"
                         for s in case['steps']]
                    )
                    # 预期结果
                    expected_result = "\n".join(
                        [f"{s['step']}. {s['expected']}\n"
                         for s in case['expected_result']]
                    )

                    rows.append({
                        "模块": crew_result['module'],
                        "*标题": case['title'],
                        "步骤描述": steps_desc_str,
                        "预期结果": expected_result,
                        "测试类型": case.get('test_type', '手动'),
                        "用例类型": case.get('case_type', '功能测试'),
                        "重要程度": case.get('priority', 'P0'),
                        "前置条件": case.get('preconditions', ''),
                        "编号": '',
                        "维护人": '',
                        "预估工时": case.get('estimated_time', ''),
                        "关联项目需求": '',
                        "关注人": '',
                        "备注": case.get('remarks', '')
                    })

                # 创建DataFrame
                df = pd.DataFrame(rows)

                # 显示表格预览
                st.dataframe(df)

                # Create Excel file in memory
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Test Cases')

                output.seek(0)

                file_name = "test_cases.xlsx"
                # Download button for Excel file
                st.download_button(
                    label="Download Excel",
                    data=output,
                    file_name="test_cases.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            else:
                st.warning("结果格式不符合预期，无法生成Excel")
                st.json(crew_result)  # 显示原始结果用于调试


if __name__ == "__main__":
    vision_page()