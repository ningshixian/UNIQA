import time
import re
import jionlp
import json

from uniqa.logging import logger, log_filter

from uniqa.lazy_imports import LazyImport
with LazyImport(message="Run 'pip install \"flashtext\"'") as flashtext_import:
    from flashtext import KeywordProcessor

# # 实体识别模块(UIE)
# from uie_pytorch.uie_predictor import UIEPredictor
# # 设定抽取目标和定制化模型权重路径
# my_ie = UIEPredictor(model='uie-base', 
#                      task_path='./car_entity_0320_1700/model_best',
#                      schema=['汽车名词'])



class SlotKeywordsNormProcessor(object):
    def __init__(self, slot_keywords_path: str):
        self.slot_keywords_processor = SlotKeywordsNormProcessor.__init_slot_keywords_processor(slot_keywords_path)

    @staticmethod
    def __load_json_dict(slot_keywords_file_path):
        """
        加载归一化关键词数据
        :param: slot_keywords_file_path: 关键词slot文件路径（json文件，key为归一化名称，value为关键词list）
        :return: dict: 归一化名称为key，关键词list为value的dict
        """
        with open(slot_keywords_file_path, 'r', encoding='utf-8') as read_file:
            return json.load(read_file)

    @staticmethod
    def __init_slot_keywords_processor(slot_keywords_file_path):
        """
        初始化归一化关键词processor
        :param: slot_keywords_file_path: 关键词slot文件路径
        :return: keyword_normalization_processor: 归一化关键词processor
        """
        if not slot_keywords_file_path:
            return None
        keyword_dict = SlotKeywordsNormProcessor.__load_json_dict(slot_keywords_file_path)
        keyword_normalization_processor = KeywordProcessor(case_sensitive=False)    # 忽略大小写
        keyword_normalization_processor.add_keywords_from_dict(keyword_dict)
        return keyword_normalization_processor

    def slot_filling(self, text):
        """
        槽位提取
        :param: text: 待抽取文本
        :return: slot_list: slot抽取结果list
        """
        return self.slot_keywords_processor.extract_keywords(text, span_info=True)


class EntityExtractor(object):
    def __init__(self):
        # 实体识别模块(关键词匹配)
        self.slot_processor = SlotKeywordsNormProcessor(qa_qy_slot_path)
        self.slot_car_processor = SlotKeywordsNormProcessor(qa_qy_slot_car_path)

    def entity_extract(self, text, entity_mode, entity):
        """根据不同实体类型提取实体
        
        entity_mode
        （0-系统/1-枚举/2-正则/3-意图/4-其他/5-算法）

        entity = {
            "entityName": "实体",
            "entityType": 0,
            "vocab": {实体值: [同义词]}
        }
        """
        flag = False    # 是否实体匹配成功
        entity_value, start_idx, end_idx = "", None, None
        entity_name = entity.get('entityName', '')
        pattern = r""
        
        if entity_mode=="prebuildEntity":   # "系统实体"
            if entity_name=="phoneNumber":  # 手机号、座机号
                pattern = r"(/^1(3[0-9]|4[01456879]|5[0-35-9]|6[2567]|7[0-8]|8[0-9]|9[0-35-9])\d{8}$/)|(/^(0\d{2,3})-?(\d{7,8})$/)"
            if entity_name=="email":
                pattern = r"/^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$/"
            if entity_name=="number":   # 对整数的提取，支持中文表示的数字和阿拉伯数字
                # 中文数字逆转 https://github.com/HaujetZhao/Chinese-ITN
                pattern = r"^[0-9]*$"
            if entity_name=="expressID":    # 快递单号
                pattern = r'\b[A-Za-z]?\d+[A-Za-z0-9]*\b'
            if entity_name=="flightNumber": # 国内航班号
                # https://blog.csdn.net/weixin_43585249/article/details/122857401
                pattern = r'[A-Z]{2}\d{1,4}'
            if entity_name=="zipcode":  # 6位邮政编码
                pattern = r"/^[1-9]\d{5}$/"
            if entity_name=="idCardNumber": # 系统.身份证号
                pattern = r"/(^\d{15}$)|(^\d{18}$)|(^\d{17}(\d|X|x)$)/"
            if entity_name=="datetime":     # 日期/时间
                # https://github.com/yiyujianghu/sinan/tree/master
                # https://github.com/dongrixinyu/JioNLP/wiki/NER-%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3#user-content-%E6%97%B6%E9%97%B4%E5%AE%9E%E4%BD%93%E6%8A%BD%E5%8F%96
                t = jionlp.ner.extract_time(text[0], time_base=time.time(), with_parsing=False)  # with_parsing是否返回解析信息
                if t and t[0]["type"] in ["time_span", "time_point"]:
                    start_idx, end_idx = t[0]["offset"]
                    entity_value = t[0]['text']
                    flag = True
            if entity_name in ["province", "city", "county"]:   # 系统.省/市/区县
                # https://gitee.com/JackerKun/worldArea/blob/master/sql/chinese_area(ID%E6%95%B4%E5%9E%8B%E5%8C%96,%E5%9F%8E%E5%B8%82%E9%A6%96%E5%AD%97%E6%AF%8D).sql
                # https://github.com/dongrixinyu/JioNLP/wiki/Gadget-%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3#user-content-%E6%96%B0%E9%97%BB%E5%9C%B0%E5%90%8D%E8%AF%86%E5%88%AB
                res = jionlp.recognize_location(text[0])
                if res and res["domestic"]:
                    entity_value = res["domestic"][0][0][entity_name]
                    start_idx, end_idx = -1, -1
                    flag = True
            if entity_name=="country":  # 系统.国家
                pass
            
            match = re.search(pattern, text[0])
            if pattern and match:
                start_idx, end_idx = match.span()
                entity_value = match.group()    # 第一个匹配的文本
                flag = True

        elif entity_mode=="enumerateEntity":    # 枚举实体
            slots = self.slot_processor.slot_filling(text[0])
            # slots = {x[0]:(x[1], x[2]) for x in slots}  # 覆盖
            # 过滤&只保留匹配【实体名称】的第一个结果
            condition = lambda item: item[0].startswith(entity_name)
            slots = [x for x in slots if condition(x)]
            if slots:
                slot = slots[0]
                entity_value = slot[0].split('-')[1]
                start_idx, end_idx = slot[1], slot[2]
                flag = True
        
        elif entity_mode=="reEntity":   # 正则实体
            try:
                pattern = next(iter(entity['vocab']))   # 获取第一个键 vocabularyEntryName
            except Exception as e:
                logger.error(f"实体pattern获取失败！{e}")
                pass
            # matches = re.finditer(pattern, text[0])   # 查找所有匹配项及其位置
            match = re.search(pattern, text[0])  # 查找第一个匹配项及其位置
            if pattern and match:
                start_idx, end_idx = match.span()
                entity_value = match.group()    # 第一个匹配的文本
                flag = True
        
        elif entity_mode=="intentEntity":   # 意图实体
            # # entity['vocab'] = {实体值: [同义词,同义词2]}
            # 构建一个包含所有实体值和同义词的正则表达式模式
            all_entities = set(entity['vocab'].keys())
            all_synonyms = {synonym for synonyms in entity['vocab'].values() for synonym in synonyms}
            all_patterns = all_entities.union(all_synonyms)
            pattern = '|'.join(map(re.escape, all_patterns))
            # matches = re.finditer(pattern, text[0])   # 查找所有匹配项及其位置
            match = re.search(pattern, text[0])  # 查找第一个匹配项及其位置
            if pattern and match:
                start_idx, end_idx = match.span()
                entity_value = match.group()    # 第一个匹配的文本
                flag = True
                # print(f"匹配到 '{entity_value}' 在位置 {start_idx}-{end_idx}")

        elif entity_mode=="otherEntity":   # 其他实体
            pass
        
        # 理想算法实体，暂仅提供一个车型提取
        elif entity_mode=="algorithmEntity":
            slots = self.slot_car_processor.slot_filling(text[0])
            if slots:
                slot = slots[0]
                entity_value, start_idx, end_idx = slot[0], slot[1], slot[2]
                flag = True
            
            # match = my_ie(text[0])[0][entity_name]
            # if match:
            #     start_idx, end_idx = match[0]['start'], match[0]['end']
            #     entity_value = match[0]['text']    # 第一个匹配的文本
            #     flag = True

        return flag, entity_value, start_idx, end_idx

