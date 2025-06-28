import csv
import json
from collections import defaultdict
import re


recognizers = [
    {
        "name": "id_card",
        "regex": r"(?<![0-9a-zA-Z_.-])([1-9][0-9]{5}(19|20)[0-9]{2}(0[1-9]|1[0-2])[0-3][0-9]{4}[0-9Xx])(?![0-9a-zA-Z_.-])",
        "score": 1.0
    },
    {
        "name": "phone",
        "regex": r"(?<![0-9a-zA-Z_.-])((13[0-9]|14[014-9]|15[0-35-9]|16[2567]|17[0-8]|18[0-9]|19[0-35-9])\d{8})(?![0-9a-zA-Z_@.|-])",
        "score": 0.7
    },
    {
        "name": "frame_number",
        "regex": r"(?<![0-9a-zA-Z_.-])([A-HJ-NPR-Z]{2}[A-HJ-NPR-Z\d]{6}[X\d][A-HJ-NPR-TV-Y\d][A-HJ-NPR-Z\d]{2}\d{5})(?![0-9a-zA-Z_.-])",
        "score": 0.9
    },
    {
        "name": "atm_card",
        "regex": r"(?<![0-9a-zA-Z_.-])(((62)\d{17})|((62)\d{14}))(?![0-9a-zA-Z_.-])",
        "score": 0.8
    },
    {
        "name": "password",
        "regex": r"(?:password|x-chj-key|websocket-key)[:：“=\"\s\\]+(?:u003d)?([\w@.:%!+?*#=/\\-]{7,})",
        "score": 1.0
    },
    {
        "name": "token",
        "regex": r"(?:token|token_prod|jwt)[:：\"=\s\\]+(?:u003d)?([\w@.:%!+?*#=/\\-]{12,})",
        "score": 1.0
    },
    {
        "name": "latitude",
        "regex": r"(?:latitude|longitude|lat|lng)[\":= ]+[+-]?((0|([1-9](\d)?)|(1[0-7]\d)|180)(\.\d{4,15}))",
        "score": 1.0
    },
    {
        "name": "email",
        "regex": r"(?<![0-9a-zA-Z_.-])([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)(?![0-9a-zA-Z_.-])",
        "score": 1.0
    },
    {
        "name": "lx",
        "regex": r'(理想|li|l\d+|i\d+|mega|one)+',
        "score": 1.0
    },
]

# 数据脱敏
def redact_sensitive_data(text):
    for recognizer in recognizers:
        name = recognizer["name"].upper()
        regex = recognizer["regex"]
        text = re.sub(regex, f"[{name}]", text, flags=re.IGNORECASE)
    
    return text

def process_csv_to_json(input_csv_path, output_json_path):
    # 读取CSV文件并按 question_id 分组
    questions_by_id = defaultdict(list)
    answers_by_id = defaultdict(str)

    # ,question_id,question_content,answer_content,base_name,car_type,source
    with open(input_csv_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question_id = row['question_id']
            question_content = row['question_content'].strip()
            answer_content = row['answer_content'].strip()
            if question_content:  # 避免空问题
                questions_by_id[question_id].append(question_content)
            if answer_content.strip():  # 避免空答案
                answers_by_id[question_id] = answer_content

    # 构建JSON结构
    result = []
    for idx, (question_id, questions) in enumerate(questions_by_id.items()):
        standard_question = redact_sensitive_data(questions[0])  # 第一个是标准问题
        similar_questions = [redact_sensitive_data(x) for x in questions[1:]]  # 剩余的是相似问题
        answer = redact_sensitive_data(answers_by_id[question_id])  # 答案留空或从其他字段提取（如果需要）
        category = "业务库"  # 示例分类，可调整
        kid = f"KB00{idx + 1}"

        result.append({
            "id": kid,
            "standard_question": standard_question,
            "similar_questions": similar_questions,
            "answer": f"【{kid}】-【{standard_question[:20]}】-【答案脱敏】",
            "category": category
        })

    # 写入JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(result, jsonfile, ensure_ascii=False, indent=2)

# 使用示例
input_csv = 'qa4api.csv'
output_json = 'qa.json'

process_csv_to_json(input_csv, output_json)
print("JSON文件转换完成")
