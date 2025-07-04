"""
解决haystack对中文不友好的问题，提供中文DocumentSplitter组件
https://zhuanlan.zhihu.com/p/1905196184680764047
https://github.com/mc112611/haystack/blob/307f8340b2e1a9104efe4e33d8c1885d17143c36/haystack/components/preprocessors/test_chinese_document_spliter.py

pip install haystack-ai == 2.12.1
pip install hanlp
"""

# import os
# import sys
# sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath("."))

from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from more_itertools import windowed
import hanlp
from uniqa import Document, logging

# from uniqa.components.preprocessors.sentence_tokenizer import Language, SentenceSplitter, nltk_imports
from uniqa.core.serialization import default_from_dict, default_to_dict

# from uniqa.utils import deserialize_callable, serialize_callable

logger = logging.logDog

Language = Literal[
    "en", "zh"
]

# mapping of split by character, 'function' and 'sentence' don't split by character
_CHARACTER_SPLIT_BY_MAPPING = {
    "page": "\f",
    "passage": "\n\n",
    "period": ".",
    "word": " ",
    "line": "\n",
}
# 加载中文的分词器
chinese_tokenizer_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
# 加载中文的句子切分器
split_sent = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)


class ChineseDocumentSpliter:
    """
    Splits long documents into smaller chunks.

    This is a common preprocessing step during indexing. It helps Embedders create meaningful semantic representations
    and prevents exceeding language model context limits.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.preprocessors import ChineseDocumentSpliter

    doc = Document(content="Moonlight shimmered softly, wolves howled nearby, night enveloped everything.")

    splitter = ChineseDocumentSpliter(
        split_by="sentence",
        split_length=10,
        split_overlap=0,
        language="zh",
        respect_sentence_boundary=False,
    )

    # splitter = ChineseDocumentSpliter(
    #     split_by="word",
    #     split_length=200,
    #     split_overlap=0,
    #     language="zh",
    #     respect_sentence_boundary=True,
    # )

    result = splitter.run(documents=[doc])
    ```
    """

    def __init__(
        self,
        split_by: Literal["function", "page", "passage", "period", "word", "sentence"] = "word",
        split_length: int = 200,
        split_overlap: int = 0,
        split_threshold: int = 0,
        splitting_function: Optional[Callable[[str], List[str]]] = None,
        respect_sentence_boundary: bool = False,
        language: Language = "zh",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
        particle_size: Literal["coarse", "fine"] = "coarse",
    ):
        """
        Initialize DocumentSplitter.

        :param split_by: 指定分割的单位。可选择：
            - `word` 按单词分割
            - `sentence` 按句子分割（使用正则逻辑或 hanlp 的句子分割器）
            - `page` 按页分割（以换页符 "\\f" 为分割依据）
            - `passage` 按段落分割（以双换行符 "\\n\\n" 为分割依据）
            - `function` 使用自定义函数进行分割
        :param split_length: 每个分块的最大单位数（例如，最大单词数或句子数）。
        :param split_overlap: 每个分块之间的重叠单位数（例如，前后分块共享的单词数）。
        :param split_threshold: 分块的最小单位数。如果一个分块的单位数少于该值，则会合并到前一个分块中。
        :param respect_sentence_boundary: 是否在按单词分割时尊重句子边界。如果为 True，将确保分割发生在句子之间。
        :param splitting_function: N当 `split_by` 设置为 "function" 时，此参数需要传入一个自定义的分割函数。
        :param language: Choose the language for the NLTK tokenizer. The default is English ("en").
        :param use_split_rules: Choose whether to use additional split rules when splitting by `sentence`.
        :param extend_abbreviations: Choose whether to extend NLTK's PunktTokenizer abbreviations with a list
            of curated abbreviations, if available. This is currently supported for English ("en") and German ("de").
        :param particle_size: 分词粒度。coarse代表粗颗粒度中文分词，fine代表细颗粒度分词
        """

        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold
        self.splitting_function = splitting_function
        self.respect_sentence_boundary = respect_sentence_boundary
        self.language = language
        self.use_split_rules = use_split_rules
        self.extend_abbreviations = extend_abbreviations

        # coarse代表粗颗粒度中文分词，fine代表细颗粒度分词，默认为粗颗粒度分词
        # 'coarse' represents coarse granularity Chinese word segmentation, 'fine' represents fine granularity word segmentation, default is coarse granularity word segmentation
        self.particle_size = particle_size

        self._init_checks(
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            splitting_function=splitting_function,
            respect_sentence_boundary=respect_sentence_boundary,
        )
        self._use_sentence_splitter = split_by == "sentence" or (respect_sentence_boundary and split_by == "word")
        self.sentence_splitter = None

    def _init_checks(
        self,
        *,
        split_by: str,
        split_length: int,
        split_overlap: int,
        splitting_function: Optional[Callable],
        respect_sentence_boundary: bool,
    ) -> None:
        """
        Validates initialization parameters for DocumentSplitter.

        :param split_by: The unit for splitting documents
        :param split_length: The maximum number of units in each split
        :param split_overlap: The number of overlapping units for each split
        :param splitting_function: Custom function for splitting when split_by="function"
        :param respect_sentence_boundary: Whether to respect sentence boundaries when splitting
        :raises ValueError: If any parameter is invalid
        """
        valid_split_by = ["function", "page", "passage", "period", "word", "line", "sentence"]
        if split_by not in valid_split_by:
            raise ValueError(f"split_by must be one of {', '.join(valid_split_by)}.")

        if split_by == "function" and splitting_function is None:
            raise ValueError("When 'split_by' is set to 'function', a valid 'splitting_function' must be provided.")

        if split_length <= 0:
            raise ValueError("split_length must be greater than 0.")

        if split_overlap < 0:
            raise ValueError("split_overlap must be greater than or equal to 0.")

        if respect_sentence_boundary and split_by != "word":
            logger.warning(
                "The 'respect_sentence_boundary' option is only supported for `split_by='word'`. "
                "The option `respect_sentence_boundary` will be set to `False`."
            )
            self.respect_sentence_boundary = False

    def warm_up(self):
        """
        Warm up the DocumentSplitter by loading the sentence tokenizer.
        """
        if self._use_sentence_splitter and self.sentence_splitter is None:
            self.sentence_splitter = self.chinese_sentence_split    # zh
    
    # @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        对文档进行分割处理。

        根据指定的分割方式（`split_by`），将文档分割为更小的部分。

        :param documents: 要分割的文档列表，每个文档是一个 `Document` 对象。

        :returns: 一个包含分割后文档的字典，结构如下：
            - `documents`: 分割后的文档列表。每个文档包含以下信息：
                - `source_id`: 原始文档的 ID，用于追踪来源。
                - 其他元数据字段保持不变。
        """
        if self._use_sentence_splitter and self.sentence_splitter is None:
            raise RuntimeError(
                "The component DocumentSplitter wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs: List[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but content for document ID {doc.id} is None."
                )
            if doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue

            split_docs += self._split_document(doc)
        return {"documents": split_docs}
    
    # 定义一个函数用于处理中文分句
    def chinese_sentence_split(self, text: str) -> list:
        # 分句
        sentences = split_sent(text)

        # 整理格式
        results = []
        start = 0
        for sentence in sentences:
            start = text.find(sentence, start)
            end = start + len(sentence)
            results.append({"sentence": sentence + "\n", "start": start, "end": end})
            start = end

        return results

    # 根据指定的 `split_by` 参数将文本分割成多个单元。
    def _split_document(self, doc: Document) -> List[Document]:
        if self.split_by == "sentence" or self.respect_sentence_boundary:
            return self._split_by_hanlp_sentence(doc)

        if self.split_by == "function" and self.splitting_function is not None:
            return self._split_by_function(doc)

        return self._split_by_character(doc)

    # 增加中文句子切分，通过languge == "zh"，进行启用
    def _split_by_hanlp_sentence(self, doc: Document) -> List[Document]:
        split_docs = []
        
        # if self.language == "zh":
        result = self.chinese_sentence_split(doc.content)
        units = [sentence["sentence"] for sentence in result]

        if self.respect_sentence_boundary:
            text_splits, splits_pages, splits_start_idxs = (
                self._concatenate_sentences_based_on_word_amount(
                    sentences=units,
                    split_length=self.split_length,
                    split_overlap=self.split_overlap,
                    language=self.language,
                    particle_size=self.particle_size,
                )
            )
        else:
            text_splits, splits_pages, splits_start_idxs = self._concatenate_units(
                elements=units,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
                split_threshold=self.split_threshold,
            )
        metadata = deepcopy(doc.meta)
        metadata["source_id"] = doc.id
        split_docs += self._create_docs_from_splits(
            text_splits=text_splits,
            splits_pages=splits_pages,
            splits_start_idxs=splits_start_idxs,
            meta=metadata,
        )
        return split_docs
    
    def _split_by_character(self, doc) -> List[Document]:
        split_at = _CHARACTER_SPLIT_BY_MAPPING[self.split_by]
        if self.split_by in ["page", "passage"]:
            units = doc.content.split(split_at)
        elif self.language == "zh" and self.particle_size == "coarse":
            units = chinese_tokenizer_coarse(doc.content)
        elif self.language == "zh" and self.particle_size == "fine":
            units = chinese_tokenizer_fine(doc.content)
        else:   
            raise NotImplementedError(
                "ChineseDocumentSplitter only supports 'function', 'page', 'passage', 'sentence', or 'word' as split units."
            )
            # Add the delimiter back to all units except the last one
        
        # 将分隔符添加回所有单元（最后一个单元除外）
        for i in range(len(units) - 1):
            units[i] += split_at
        
        text_splits, splits_pages, splits_start_idxs = self._concatenate_units(
            units, self.split_length, self.split_overlap, self.split_threshold
        )
        metadata = deepcopy(doc.meta)
        metadata["source_id"] = doc.id
        return self._create_docs_from_splits(
            text_splits=text_splits,
            splits_pages=splits_pages,
            splits_start_idxs=splits_start_idxs,
            meta=metadata,
        )
    
    def _split_by_function(self, doc) -> List[Document]:
        # the check for None is done already in the run method
        splits = self.splitting_function(doc.content)  # type: ignore
        docs: List[Document] = []
        for s in splits:
            meta = deepcopy(doc.meta)
            meta["source_id"] = doc.id
            docs.append(Document(content=s, meta=meta))
        return docs

    def _concatenate_units(
        self,
        elements: List[str],
        split_length: int,
        split_overlap: int,
        split_threshold: int,
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        将元素连接成长度为 split_length 单位的部分。
        
        同时记录每个元素所属的原始页码。如果当前单位的长度小于预定义的 split_threshold，则不会创建新的拆分。
        相反，它会将当前单位与上一个拆分连接起来，以避免产生过小的拆分。
        """

        text_splits: List[str] = []
        splits_pages: List[int] = []
        splits_start_idxs: List[int] = []
        cur_start_idx = 0
        cur_page = 1
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)

        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)

            # check if length of current units is below split_threshold
            if len(current_units) < split_threshold and len(text_splits) > 0:
                # concatenate the last split with the current one
                text_splits[-1] += txt

            # NOTE: This line skips documents that have content=""
            elif len(txt) > 0:
                text_splits.append(txt)
                splits_pages.append(cur_page)
                splits_start_idxs.append(cur_start_idx)

            processed_units = current_units[: split_length - split_overlap]
            cur_start_idx += len("".join(processed_units))

            if self.split_by == "page":
                num_page_breaks = len(processed_units)
            else:
                num_page_breaks = sum(
                    processed_unit.count("\f") for processed_unit in processed_units
                )

            cur_page += num_page_breaks

        return text_splits, splits_pages, splits_start_idxs

    def _create_docs_from_splits(
        self,
        text_splits: List[str],
        splits_pages: List[int],
        splits_start_idxs: List[int],
        meta: Dict[str, Any],
    ) -> List[Document]:
        """
        Creates Document objects from splits enriching them with page number and the metadata of the original document.
        """
        documents: List[Document] = []

        for i, (txt, split_idx) in enumerate(zip(text_splits, splits_start_idxs)):
            copied_meta = deepcopy(meta)
            copied_meta["page_number"] = splits_pages[i]
            copied_meta["split_id"] = i
            copied_meta["split_idx_start"] = split_idx
            doc = Document(content=txt, meta=copied_meta)
            documents.append(doc)

            if self.split_overlap <= 0:
                continue

            doc.meta["_split_overlap"] = []

            if i == 0:
                continue

            doc_start_idx = splits_start_idxs[i]
            previous_doc = documents[i - 1]
            previous_doc_start_idx = splits_start_idxs[i - 1]
            self._add_split_overlap_information(
                doc, doc_start_idx, previous_doc, previous_doc_start_idx
            )

        for d in documents:
            d.content = d.content.replace(" ", "")
        return documents

    @staticmethod
    def _add_split_overlap_information(
        current_doc: Document,
        current_doc_start_idx: int,
        previous_doc: Document,
        previous_doc_start_idx: int,
    ):
        """
        Adds split overlap information to the current and previous Document's meta.

        :param current_doc: The Document that is being split.
        :param current_doc_start_idx: The starting index of the current Document.
        :param previous_doc: The Document that was split before the current Document.
        :param previous_doc_start_idx: The starting index of the previous Document.
        """
        overlapping_range = (
            current_doc_start_idx - previous_doc_start_idx,
            len(previous_doc.content),
        )  # type: ignore

        if overlapping_range[0] < overlapping_range[1]:
            # type: ignore
            overlapping_str = previous_doc.content[
                overlapping_range[0] : overlapping_range[1]
            ]

            if current_doc.content.startswith(overlapping_str):  # type: ignore
                # add split overlap information to this Document regarding the previous Document
                current_doc.meta["_split_overlap"].append(
                    {"doc_id": previous_doc.id, "range": overlapping_range}
                )

                # add split overlap information to previous Document regarding this Document
                overlapping_range = (0, overlapping_range[1] - overlapping_range[0])
                previous_doc.meta["_split_overlap"].append(
                    {"doc_id": current_doc.id, "range": overlapping_range}
                )

    # def to_dict(self) -> Dict[str, Any]:
    #     """
    #     Serializes the component to a dictionary.
    #     """
    #     serialized = default_to_dict(
    #         self,
    #         split_by=self.split_by,
    #         split_length=self.split_length,
    #         split_overlap=self.split_overlap,
    #         split_threshold=self.split_threshold,
    #         respect_sentence_boundary=self.respect_sentence_boundary,
    #         language=self.language,
    #         use_split_rules=self.use_split_rules,
    #         extend_abbreviations=self.extend_abbreviations,
    #     )
    #     if self.splitting_function:
    #         serialized["init_parameters"]["splitting_function"] = serialize_callable(self.splitting_function)
    #     return serialized

    # @classmethod
    # def from_dict(cls, data: Dict[str, Any]) -> "DocumentSplitter":
    #     """
    #     Deserializes the component from a dictionary.
    #     """
    #     init_params = data.get("init_parameters", {})

    #     splitting_function = init_params.get("splitting_function", None)
    #     if splitting_function:
    #         init_params["splitting_function"] = deserialize_callable(splitting_function)

    #     return default_from_dict(cls, data)
    

    @staticmethod
    def _concatenate_sentences_based_on_word_amount(
        sentences: List[str],
        split_length: int,
        split_overlap: int,
        language: str,
        particle_size: str,
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Groups the sentences into chunks of `split_length` words while respecting sentence boundaries.

        This function is only used when splitting by `word` and `respect_sentence_boundary` is set to `True`, i.e.:
        with NLTK sentence tokenizer.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: A tuple containing the concatenated sentences, the start page numbers, and the start indices.
        """
        # chunk information
        chunk_word_count = 0
        chunk_starting_page_number = 1
        chunk_start_idx = 0
        current_chunk: List[str] = []
        # output lists
        split_start_page_numbers = []
        list_of_splits: List[List[str]] = []
        split_start_indices = []
        # chinese_tokenizer_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        # chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        for sentence_idx, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if language == "zh" and particle_size == "coarse":
                chunk_word_count += len(chinese_tokenizer_coarse(sentence))
                next_sentence_word_count = (
                    len(chinese_tokenizer_coarse(sentences[sentence_idx + 1]))
                    if sentence_idx < len(sentences) - 1
                    else 0
                )
            if language == "zh" and particle_size == "fine":
                chunk_word_count += len(chinese_tokenizer_fine(sentence))
                next_sentence_word_count = (
                    len(chinese_tokenizer_fine(sentences[sentence_idx + 1]))
                    if sentence_idx < len(sentences) - 1
                    else 0
                )

            # Number of words in the current chunk plus the next sentence is larger than the split_length,
            # or we reached the last sentence
            if (
                chunk_word_count + next_sentence_word_count
            ) > split_length or sentence_idx == len(sentences) - 1:
                #  Save current chunk and start a new one
                list_of_splits.append(current_chunk)
                split_start_page_numbers.append(chunk_starting_page_number)
                split_start_indices.append(chunk_start_idx)

                # Get the number of sentences that overlap with the next chunk
                num_sentences_to_keep = (
                    ChineseDocumentSpliter._number_of_sentences_to_keep(
                        sentences=current_chunk,
                        split_length=split_length,
                        split_overlap=split_overlap,
                        language=language,
                        particle_size=particle_size,
                    )
                )
                # Set up information for the new chunk
                if num_sentences_to_keep > 0:
                    # Processed sentences are the ones that are not overlapping with the next chunk
                    processed_sentences = current_chunk[:-num_sentences_to_keep]
                    chunk_starting_page_number += sum(
                        sent.count("\f") for sent in processed_sentences
                    )
                    chunk_start_idx += len("".join(processed_sentences))
                    # Next chunk starts with the sentences that were overlapping with the previous chunk
                    current_chunk = current_chunk[-num_sentences_to_keep:]
                    chunk_word_count = sum(len(s.split()) for s in current_chunk)
                else:
                    # Here processed_sentences is the same as current_chunk since there is no overlap
                    chunk_starting_page_number += sum(
                        sent.count("\f") for sent in current_chunk
                    )
                    chunk_start_idx += len("".join(current_chunk))
                    current_chunk = []
                    chunk_word_count = 0

        # Concatenate the sentences together within each split
        text_splits = []
        for split in list_of_splits:
            text = "".join(split)
            if len(text) > 0:
                text_splits.append(text)

        return text_splits, split_start_page_numbers, split_start_indices

    @staticmethod
    def _number_of_sentences_to_keep(
        sentences: List[str],
        split_length: int,
        split_overlap: int,
        language: str,
        particle_size: str,
    ) -> int:
        """
        Returns the number of sentences to keep in the next chunk based on the `split_overlap` and `split_length`.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: The number of sentences to keep in the next chunk.
        """
        # If the split_overlap is 0, we don't need to keep any sentences
        if split_overlap == 0:
            return 0

        num_sentences_to_keep = 0
        num_words = 0
        # chinese_tokenizer_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        # chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        # Next overlapping Document should not start exactly the same as the previous one, so we skip the first sentence
        for sent in reversed(sentences[1:]):
            if language == "zh" and particle_size == "coarse":
                num_words += len(chinese_tokenizer_coarse(sent))
                # num_words += len(sent.split())
            if language == "zh" and particle_size == "fine":
                num_words += len(chinese_tokenizer_fine(sent))
            # If the number of words is larger than the split_length then don't add any more sentences
            if num_words > split_length:
                break
            num_sentences_to_keep += 1
            if num_words > split_overlap:
                break
        return num_sentences_to_keep


if __name__ == "__main__":

    from pprint import pprint as print

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
                                万物静候，聆听着夜的呼吸！"""
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

    # # 以sentence进行切分时，切分长度为2个句子，重叠窗口为上下文有一个句子为重叠
    # splitter = ChineseDocumentSpliter(
    #     split_by="sentence",
    #     split_length=2,
    #     split_overlap=1,
    #     language="zh",
    #     respect_sentence_boundary=False,
    # )
    # splitter.warm_up()
    # result = splitter.run(documents=[doc])
    # for d in result["documents"]:
    #     print("====>" + d.content)

    # 以sentence进行切分时，切分长度为2个句子，重叠窗口为上下文有一个句子为重叠
    splitter = ChineseDocumentSpliter(
        split_by="sentence",
        split_length=1,
        split_overlap=0,
        language="zh",
        respect_sentence_boundary=False,
    )
    splitter.warm_up()
    result = splitter.run(documents=[doc])
    for d in result["documents"]:
        print("====>" + d.content)
