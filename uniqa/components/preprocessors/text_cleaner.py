# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
import string
from typing import Any, Dict, List, Optional

# from uniqa import component


# @component
class TextCleaner:
    """
    Cleans text strings.

    It can remove substrings matching a list of regular expressions, convert text to lowercase,
    remove punctuation, and remove numbers.
    Use it to clean up text data before evaluation.

    ### Usage example

    ```python
    from haystack.components.preprocessors import TextCleaner

    text_to_clean = "1Moonlight shimmered softly, 300 Wolves howled nearby, Night enveloped everything."

    cleaner = TextCleaner(convert_to_lowercase=True, remove_punctuation=False, remove_numbers=True)
    result = cleaner.run(texts=[text_to_clean])
    ```
    """

    def __init__(
        self,
        remove_regexps: Optional[List[str]] = None,
        convert_to_lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_emoji: bool = False,
        http_normalization: bool = False,
        phone_normalization: bool = False,
        time_normalization: bool = False,
    ):
        """
        Initializes the TextCleaner component.

        :param remove_regexps: A list of regex patterns to remove matching substrings from the text.
        :param convert_to_lowercase: If `True`, converts all characters to lowercase.
        :param remove_punctuation: If `True`, removes punctuation from the text.
        :param remove_numbers: If `True`, removes numerical digits from the text.
        """
        self._remove_regexps = remove_regexps
        self._convert_to_lowercase = convert_to_lowercase
        self._remove_punctuation = remove_punctuation
        self._remove_numbers = remove_numbers
        self._remove_emoji = remove_emoji
        self._http_normalization = http_normalization
        self._phone_normalization = phone_normalization
        self._time_normalization = time_normalization

        self._regex = None
        if remove_regexps:
            self._regex = re.compile("|".join(remove_regexps), flags=re.IGNORECASE)
        to_remove = ""
        if remove_punctuation:
            to_remove = string.punctuation
            to_remove += "[’!\"#$%&'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+"  # 中文标点符号
        if remove_numbers:
            to_remove += string.digits
        if remove_emoji:
            to_remove += u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+'

        self._translator = str.maketrans("", "", to_remove) if to_remove else None

    # 时间归一化
    def time_normalization_func(self, text):
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
    
    def http_normalization_func(self, text):
        # http替换为[HTTP]
        text = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", '[HTTP]', text).replace('\t','').replace('&nbsp','').strip()
        return text
    
    # 时间归一化
    def phone_normalization_func(self, text):
        # 手机号码替换为[PHONE]
        pattern = r"(?:^|[^\d])((?:\+?86)?1(?:3\d{3}|5[^4\D]\d{2}|8\d{3}|7(?:[01356789]\d{2}|4(?:0\d|1[0-2]|9\d))|9[189]\d{2}|6[567]\d{2}|4(?:[14]0\d{1}|[68]\d{2}|[579]\d{2}))\d{6})(?:$|[^\d])"
        phone_list = re.compile(pattern).findall(text)
        for phone_number in phone_list:
            text = re.sub(phone_number, '[PHONE]', text)
            # text = re.sub(repr(phone_number), '[PHONE]', text)
        # text = re.sub(r"(?:^|\D)?(?:\+?86)?1(?:3\d{3}|5[^4\D]\d{2}|8\d{3}|7(?:[01356789]\d{2}|4(?:0\d|1[0-2]|9\d))|9[189]\d{2}|6[567]\d{2}|4(?:[14]0\d{1}|[68]\d{2}|[579]\d{2}))\d{6}(?:^|\D)?", '[PHONE]', text)
        text = re.sub("1(\d{2})((\*){4})(\d{4})", '[PHONE]', text)  # 135****4934

        # # 尾号替换为[SUBPN]
        # text = re.sub("尾号.?(\d{4})", '尾号[SUBPN]', text)
        # text = re.sub("(\d{4}).?尾号", '[SUBPN]尾号', text)
        return text

    # @component.output_types(texts=List[str])
    def run(self, texts: List[str]) -> Dict[str, Any]:
        """
        Cleans up the given list of strings.

        :param texts: List of strings to clean.
        :returns: A dictionary with the following key:
            - `texts`:  the cleaned list of strings.
        """

        if self._time_normalization:
            texts = [self.time_normalization_func(text) for text in texts]
        
        if self._http_normalization:
            texts = [self.http_normalization_func(text) for text in texts]
        
        if self._phone_normalization:
            texts = [self.phone_normalization_func(text) for text in texts]

        if self._regex:
            texts = [self._regex.sub("", text) for text in texts]

        if self._convert_to_lowercase:
            texts = [text.lower() for text in texts]

        if self._translator:
            texts = [text.translate(self._translator) for text in texts]
        
        return {"texts": texts}
