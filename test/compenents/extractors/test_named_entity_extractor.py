# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Note: We do not test the Spacy backend in this module.
# Spacy is not installed in the test environment to keep the CI fast.
# We test the Spacy backend in e2e/pipelines/test_named_entity_extractor.py.

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

import pytest

from uniqa import Document
from uniqa.components.extractors import NamedEntityAnnotation, NamedEntityExtractor, NamedEntityExtractorBackend
from uniqa.utils.device import ComponentDevice

"""
pytest -svx test/test_named_entity_extractor.py::test_ner_extractor_init
pytest -svx test/test_named_entity_extractor.py::test_ner_extractor_spacy_backend
"""


documents = [
    Document(content="理想L9这款车是由中国的理想汽车在 2021年 生产的"),
    Document(content="台湾是一个位于亚洲东部的岛屿国家。"),
]

extractor = NamedEntityExtractor(
    backend=NamedEntityExtractorBackend.SPACY, 
    model="zh_core_web_trf", 
    device=ComponentDevice.from_str("mps"),
)
extractor.warm_up()
results = extractor.run(documents=documents)["documents"]
annotations = [NamedEntityExtractor.get_stored_annotations(doc) for doc in results]  # doc["named_entities"]
print(annotations)

"""
[
    [NamedEntityAnnotation(entity='PRODUCT', start=0, end=4, score=None), 
    NamedEntityAnnotation(entity='GPE', start=9, end=11, score=None), 
    NamedEntityAnnotation(entity='ORG', start=12, end=18, score=None), 
    NamedEntityAnnotation(entity='DATE', start=20, end=25, score=None)], 
    
    [NamedEntityAnnotation(entity='GPE', start=0, end=2, score=None), 
    NamedEntityAnnotation(entity='LOC', start=7, end=9, score=None)]
]
"""


@pytest.fixture
def raw_texts():
    return [
        "理想L9这款车是由中国的理想汽车公司在 2021年 生产的",
        "",  # Intentionally empty.
        "台湾是一个位于亚洲东部的岛屿国家。"
    ]


# @pytest.fixture
# def hf_annotations():
#     return [
#         [
#             NamedEntityAnnotation(entity="PER", start=11, end=16),
#             NamedEntityAnnotation(entity="LOC", start=31, end=39),
#             NamedEntityAnnotation(entity="LOC", start=41, end=51),
#         ],
#         [], 
#         [
#             NamedEntityAnnotation(entity='LOC', start=0, end=2, score=None), 
#             NamedEntityAnnotation(entity='LOC', start=7, end=9, score=None)
#         ],
#     ]


@pytest.fixture
def spacy_annotations():
    return [
        [
            NamedEntityAnnotation(entity='PRODUCT', start=0, end=4, score=None), 
            NamedEntityAnnotation(entity='GPE', start=9, end=11, score=None), 
            NamedEntityAnnotation(entity='ORG', start=12, end=18, score=None), 
            NamedEntityAnnotation(entity='DATE', start=20, end=25, score=None),
        ],
        [], 
        [
            NamedEntityAnnotation(entity='GPE', start=0, end=2, score=None), 
            NamedEntityAnnotation(entity='LOC', start=7, end=9, score=None)
        ],
    ]


def test_ner_extractor_init():
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.SPACY, model="zh_core_web_trf")

    with pytest.raises(RuntimeError, match=r"not warmed up"):
        extractor.run(documents=[])

    assert not extractor.initialized
    extractor.warm_up()
    assert extractor.initialized


@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_spacy_backend(raw_texts, spacy_annotations, batch_size):
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.SPACY, model="zh_core_web_trf")
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, spacy_annotations, batch_size)


# @pytest.mark.parametrize("batch_size", [1, 3])
# def test_ner_extractor_hf_backend(raw_texts, hf_annotations, batch_size):
#     extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER")
#     extractor.warm_up()

#     _extract_and_check_predictions(extractor, raw_texts, hf_annotations, batch_size)


# @pytest.mark.parametrize("batch_size", [1, 3])
# @pytest.mark.skipif(
#     not os.environ.get("HF_API_TOKEN", None),
#     reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
# )
# def test_ner_extractor_hf_backend_private_models(raw_texts, hf_annotations, batch_size):
#     extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="deepset/bert-base-NER")
#     extractor.warm_up()

#     _extract_and_check_predictions(extractor, raw_texts, hf_annotations, batch_size)


# @pytest.mark.parametrize("batch_size", [1, 3])
# def test_ner_extractor_in_pipeline(raw_texts, hf_annotations, batch_size):
#     pipeline = Pipeline()
#     pipeline.add_component(
#         name="ner_extractor",
#         instance=NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER"),
#     )

#     outputs = pipeline.run(
#         {"ner_extractor": {"documents": [Document(content=text) for text in raw_texts], "batch_size": batch_size}}
#     )["ner_extractor"]["documents"]
#     predicted = [NamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]
#     _check_predictions(predicted, hf_annotations)


def _extract_and_check_predictions(extractor, texts, expected, batch_size):
    docs = [Document(content=text) for text in texts]
    outputs = extractor.run(documents=docs, batch_size=batch_size)["documents"]
    assert all(id(a) == id(b) for a, b in zip(docs, outputs))
    predicted = [NamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]
    print(predicted)

    _check_predictions(predicted, expected)


def _check_predictions(predicted, expected):
    assert len(predicted) == len(expected)
    for pred, exp in zip(predicted, expected):
        assert len(pred) == len(exp)

        for a, b in zip(pred, exp):
            assert a.entity == b.entity
            assert a.start == b.start
            assert a.end == b.end

