import re
import html

"""
- 数据清洗脚本
  - 包括：去除HTML标签；去除头尾的双引号；去除颜文字；去除【级别：二级。简介：】；手机号码替换为[PHONE]；时间替换为[TIME]；...
"""

def time_ext(text):
    # 时间归一化
    # 仅针对 年月日时分秒
    year = r"20\d{2}"
    month = r"(1[0-2]|[0]?[1-9])"
    day = r"([1-2][0-9]|3[01]|[0]?[1-9])"
    hour = r"([0]?[0-9]|1[0-9]|2[0-4])"
    minute = r"[0-5][0-9]"
    second = r"[0-5][0-9]"

    # text = re.sub(r"(2\d{3}[-\.]\d{1,2}[-\.]\d{1,2}\s[0-2]?[0-9]:[0-5][0-9]:[0-5][0-9])", "[TIME]", text)
    # text = re.sub(r"(2\d{3}[-\.]\d{1,2}[-\.]\d{1,2}\s[0-2]?[0-9]:[0-5][0-9])", "[TIME]", text)
    # text = re.sub(r"(2\d{3}[-\.]\d{1,2}[-\.]\d{1,2})", "[TIME]", text)
    # text = re.sub(r"(2\d{3})[年-]\d{1,2}[月-]\d{1,2}[日号][0-2]?[0-9]:[0-5][0-9]:[0-5][0-9]", '[TIME]', text)   # [年/-]
    # text = re.sub(r"(2\d{3})[年-]\d{1,2}[月-]\d{1,2}[日号][0-2]?[0-9]:[0-5][0-9]", '[TIME]', text)
    # text = re.sub(r"(2\d{3})[年-]\d{1,2}[月-]\d{1,2}[日号]?", '[TIME]', text)

    tmp1 = r"({})?".format(r"[-~]" + r"({})?".format(year + r"[-\.]") + month + r"[-\.]" + day)  # xxxx.mm.dd-mm.dd
    tmp2 = r"({})".format(r"[-~]" + hour + r"[:：]" + minute)   # hh:mm-hh:mm
    tmp3 = r"({})".format(hour + r"[时点]" + r"({})?".format(minute + r"分") + r"({})?".format(second + r"秒"))   # xx时xx分xx秒
    text = re.sub(year + r"[-\.]" + month + r"[-\.]" + day + r"\s" + hour + r"[:：]" + minute + r"[-~]" + hour + r"[:：]" + minute, "[TIME]", text)
    text = re.sub(year + r"[-\.]" + month + r"[-\.]" + day + tmp1 + r"\s" + hour + r"[:：]" + minute + tmp2, "[TIME]", text)
    text = re.sub(year + r"[-\.]" + month + r"[-\.]" + day + r"\s" + hour + r"[:：]" + minute + r"[:：]" + second, "[TIME]", text)
    text = re.sub(year + r"[-\.]" + month + r"[-\.]" + day + r"\s" + hour + r"[:：]" + minute, "[TIME]", text)
    text = re.sub(year + r"[-\.]" + month + r"[-\.]" + day + r"\s" + tmp3, "[TIME]", text)
    text = re.sub(year + r"[-\.]" + month + r"[-\.]" + day, "[TIME]", text)
    tmp1 = r"({})?".format(r"[-~]" + r"({})?".format(year + r"[年]") + month + r"[月]" + day + r"[日号]")  # xxxx年mm月dd日-mm月dd日
    text = re.sub(year + r"[年]" + month + r"[月]" + day + r"[日号]" + r".{0,3}" + hour + r"[:：]" + minute + r"[:：]" + second, "[TIME]", text)
    text = re.sub(year + r"[年]" + month + r"[月]" + day + r"[日号]" + tmp1 + r".{0,3}" + hour + r"[:：]" + minute + tmp2, "[TIME]", text)
    text = re.sub(year + r"[年]" + month + r"[月]" + day + r"[日号]" + r".{0,3}" + hour + r"[:：]" + minute, "[TIME]", text)
    text = re.sub(year + r"[年]" + month + r"[月]" + day + r"[日号]" + r".{0,3}" + tmp3, "[TIME]", text)
    text = re.sub(year + r"[年]" + month + r"[月]" + day + r"[日号]", "[TIME]", text)

    text = re.sub(year + r"[年]" + month + r"(月份|月)", "[TIME]", text)
    return text


def emoji_ext(s, restr=''):  
    #过滤表情   
    try:  
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')  
    except re.error:  
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')  
    return co.sub(restr, s)


def punctuation_ext(text):
    # 去掉标点符号
    text = re.sub("[’!\"#$%&'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+", " ", text)
    # 去除不可见字符
    text = re.sub(
        "[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+",
        "",
        text,
    )
    return text


def get_stopword(file_path):
    # 从文件导入停用词表
    with open(file_path, "r", "utf-8") as f:
        stpwrd_content = f.read()
        stpwrdlst = stpwrd_content.splitlines()
    return stpwrdlst


def is_number(s):
    '''
    判断字符串是否为数字
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def clean_text(s: str):
    """
    清洗文本
    1.去除HTML标签
    2.去除头尾的双引号
    3.去除【级别：二级。简介|简述】【流程.】
    4.手机号码替换为[PHONE]
    5.尾号替换为[SUBPN]
    6.http替换为[HTTP]
    7.时间替换为[TIME]
    """
    s = s.strip()
    if not s: return s
    # # 去除任何空白字符
    # s = re.sub(u"\s+", "", s, flags=re.U)
    # 去除HTML标签
    s = html.unescape(s)
    s = re.sub(r'<[^>]*>', '', s)
    # 去除头尾的双引号
    s = s.strip('"').strip('\"')
    s = re.sub(r'^"|"$', '', s)
    # # 去除颜文字
    # s = emoji_ext(s)
    # 手机号码替换为[PHONE]
    pattern = r"(?:^|[^\d])((?:\+?86)?1(?:3\d{3}|5[^4\D]\d{2}|8\d{3}|7(?:[01356789]\d{2}|4(?:0\d|1[0-2]|9\d))|9[189]\d{2}|6[567]\d{2}|4(?:[14]0\d{1}|[68]\d{2}|[579]\d{2}))\d{6})(?:$|[^\d])"
    phone_list = re.compile(pattern).findall(s)
    for phone_number in phone_list:
        s = re.sub(repr(phone_number), '[PHONE]', s)
    # text = re.sub(r"(?:^|\D)?(?:\+?86)?1(?:3\d{3}|5[^4\D]\d{2}|8\d{3}|7(?:[01356789]\d{2}|4(?:0\d|1[0-2]|9\d))|9[189]\d{2}|6[567]\d{2}|4(?:[14]0\d{1}|[68]\d{2}|[579]\d{2}))\d{6}(?:^|\D)?", '[PHONE]', text)
    s = re.sub("1(\d{2})((\*){4})(\d{4})", '[PHONE]', s)  # 135****4934
    # 尾号替换为[SUBPN]
    s = re.sub("尾号.?(\d{4})", '尾号[SUBPN]', s)
    s = re.sub("(\d{4}).?尾号", '[SUBPN]尾号', s)
    # http替换为[HTTP]
    s = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", '[HTTP]', s).replace('\t','').replace('&nbsp','').strip()
    # 时间替换为[TIME]
    s = time_ext(s)
    # 去除【级别：二级。简介|简述】【流程.】
    s = re.sub(r"(^级别[:：].{2}[。 ]{1,3}.{2}[:：])", '', s)
    s = re.sub(r"(^级别[:：].级[。 ])", '', s)
    s = re.sub(r"(^流程.*简.[:：])", '', s)
    return s.strip()


def clean_text_4_sql(s: str):
    # 数据清洗统一入口
    if not s.strip(): return s
    # # 去除任何空白字符
    # s = re.sub(u"\s+", "", s, flags=re.U)
    # 去除HTML标签
    s = html.unescape(s)
    s = re.sub(r'<[^>]*>', '', s)
    # 去除头尾的双引号
    s = s.strip('"').strip('\"')
    s = re.sub(r'^"|"$', '', s)
    return s.strip()


# 打乱数据顺序
def shuffle_in_unison(x):
    import numpy as np
    shuffled_x = [0] * len(x)
    permutation = np.random.permutation(len(x))
    for old_index, new_index in enumerate(permutation):
        shuffled_x[new_index] = x[old_index]
    return shuffled_x


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


# # 示例文本
# sample_text = """
# 我的身份证是110101199003072316，电话是13812345678，
# 邮箱为example@example.com，登录密码是Pass1234@abc，
# token是abcd1234-5678-efgh-90ij，
# 理想 l9
# 车架号是AB1234567XZ890123，经纬度是lat: 39.9042，lng: 116.4074。
# """
# redacted_text = redact_sensitive_data(sample_text)
# print(redacted_text)
