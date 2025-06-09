import pandas as pd
import json


output_file = "data/converted_data_sql_train.json"

# 读取CSV文件
df = pd.read_csv('data/converted_data.csv')

# 创建空的列表来存储结果
results = []

# 遍历每一行，解析conversations列
for index, row in df.iterrows():
    # 获取conversations列的值
    conversations = row['conversations']

    # 将字符串格式的JSON转换为Python对象
    conversation_list = json.loads(conversations.replace('""', '"'))  # 替换双引号

    # 初始化问题和答案
    question = None
    answer = None

    # 遍历每个字典，找到所需的值
    for item in conversation_list:
        if item['from'] == 'human':
            question = item['value']
        elif item['from'] == 'assistant':
            answer = item['value']

    # 将结果添加到列表中
    results.append({'instruction': question, 'input': '', 'output': answer})

# 保存为JSON文件（最外层是列表）
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"转换完成，数据已保存为 {output_file}")
