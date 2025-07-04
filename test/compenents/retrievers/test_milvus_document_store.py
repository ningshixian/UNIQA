# https://github.com/milvus-io/milvus-haystack/tree/main/tests
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

import logging
from typing import List

import numpy as np
import pytest
from uniqa import Document
from uniqa.document_stores.types import DocumentStore
# from uniqa.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest

from uniqa.document_stores.milvus import MilvusDocumentStore
from uniqa.dataclasses.sparse_embedding import SparseEmbedding


logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_ARGS = {
    # "uri": "http://localhost:19530",  # This uri works for Milvus docker service
    "uri": "./milvus_test.db",  # This uri works for Milvus Lite
}

def l2_normalization(x: List[float]) -> List[float]:
    v = np.array(x)
    l2_norm = np.linalg.norm(v)
    if l2_norm == 0:
        return np.zeros_like(v)
    normalized_v = v / l2_norm
    return normalized_v.tolist()


documents = []
for i in range(3):
    doc = Document(
        content="A Foo Document",
        meta={
            "name": f"name_{i}",
            "page": "100",
            "chapter": "intro",
            "number": i+1,
            "date": "1969-07-21T20:17:40",
        },
        embedding=l2_normalization([0.5] * 64),
        sparse_embedding=SparseEmbedding(indices=[0, 1, 2 + i], values=[1.0, 2.0, 3.0]),
    )
    documents.append(doc)

document_store = MilvusDocumentStore(
    connection_args=DEFAULT_CONNECTION_ARGS,
    consistency_level="Strong",
    drop_old=True,
)

return_value = document_store.write_documents(documents)
assert document_store.count_documents() == 3
assert return_value == 3

# Test filter_documents()
filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.number", "operator": ">=", "value": 3},
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.chapter", "operator": "in", "value": ["economy", "intro"]},
                {"field": "meta.page", "operator": "==", "value": "100"},
            ],
        },
    ],
}
doc_list = document_store.filter_documents(filters)

# Test delete_documents() normal behaviour.
for doc in doc_list:
    document_store.delete_documents([doc.id])
assert document_store.count_documents() == 2


# def test_to_and_from_dict(document_store: MilvusDocumentStore):
document_store_dict = document_store.to_dict()
expected_dict = {
    "type": "uniqa.document_stores.milvus.milvus_document_store.MilvusDocumentStore",
    "init_parameters": {
        "collection_name": "uniqaCollection",
        "collection_description": "",
        "collection_properties": None,
        "connection_args": DEFAULT_CONNECTION_ARGS,
        "consistency_level": "Strong",
        "index_params": {'metric_type': 'L2', 'index_type': 'AUTOINDEX', 'params': {}},     # 
        "search_params": {'metric_type': 'L2', 'params': {}},       # 
        "drop_old": True,
        "primary_field": "id",
        "text_field": "text",
        "vector_field": "vector",
        "sparse_vector_field": None,
        "sparse_index_params": None,
        "sparse_search_params": None,
        "builtin_function": [],
        "partition_key_field": None,
        "partition_names": None,
        "replica_number": 1,
        "timeout": None,
    },
}
assert document_store_dict == expected_dict
reconstructed_document_store = MilvusDocumentStore.from_dict(document_store_dict)
# for field in vars(reconstructed_document_store):
#     if field.startswith("__") or field in ["alias", "_milvus_client"]:
#         continue
#     if field == "builtin_function":
#         for func, func_reconstructed in zip(
#             getattr(document_store, field),
#             getattr(reconstructed_document_store, field),
#         ):
#             for k, v in func.to_dict().items():
#                 if k == "function_name":
#                     continue
#                 assert v == func_reconstructed.to_dict()[k]
#     else:
#         assert getattr(reconstructed_document_store, field) == getattr(document_store, field)

