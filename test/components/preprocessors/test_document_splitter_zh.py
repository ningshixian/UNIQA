# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from typing import List
import re
import pytest
from pprint import pprint as print

from uniqa import Document
from uniqa.components.preprocessors import ChineseDocumentSpliter
# from uniqa.utils import deserialize_callable, serialize_callable


doc = Document(
    content="""月光轻轻洒落，林中传来阵阵狼嚎，夜色悄然笼罩一切。
                            树叶在微风中沙沙作响，影子在地面上摇曳不定。
                            一只猫头鹰静静地眨了眨眼，从枝头注视着四周……
                            远处的小溪哗啦啦地流淌，仿佛在向石头倾诉着什么。
                            “咔嚓”一声，某处的树枝突然断裂，然后恢复了寂静。
                            空气中弥漫着松树与湿土的气息，令人心安。
                            一只狐狸悄然出现，又迅速消失在灌木丛中。
                            天上的星星闪烁着，仿佛在诉说古老的故事。
                            时间仿佛停滞了……
                            万物静候，聆听着夜的呼吸！
                            \n\n怎么去拥有一道彩虹。怎么去拥抱一夏天的风。"""
)

# 以word进行切分，30个词为一部分，重叠部分3个词，同时尊重句子边界
splitter = ChineseDocumentSpliter(
    split_by="word",
    split_length=30,
    split_overlap=3,
    language="zh",
    respect_sentence_boundary=True,  # 开启，保证句子完整性
)
splitter.warm_up()
result = splitter.run(documents=[doc])
for d in result["documents"]:
    print("---->" + d.content)

# 以sentence进行切分时，切分长度为2个句子，重叠窗口为上下文有一个句子为重叠
for split_by in ["sentence", "passage", "page"]:
    splitter = ChineseDocumentSpliter(
        split_by=split_by,
        split_length=1,
        split_overlap=0,
        language="zh",
        respect_sentence_boundary=False,
    )
    splitter.warm_up()
    result = splitter.run(documents=[doc])
    for d in result["documents"]:
        print("====>" + d.content)


