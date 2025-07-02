# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from uniqa.components.preprocessors import TextCleaner


text_to_clean = "1Moonlight shimmered softly, 300 Wolves howled nearby, Night enveloped everything. http://www.example.com. 13553520257 2025年7月1号"

cleaner = TextCleaner(
    convert_to_lowercase=True, 
    remove_punctuation=False, 
    remove_numbers=True,
    http_normalization=True,
    phone_normalization=True,
    time_normalization=True
)
result = cleaner.run(texts=[text_to_clean])
print(result["texts"])


def test_init_default():
    cleaner = TextCleaner()
    assert cleaner._remove_regexps is None
    assert not cleaner._convert_to_lowercase
    assert not cleaner._remove_punctuation
    assert not cleaner._remove_numbers
    assert cleaner._regex is None
    assert cleaner._translator is None


def test_run():
    cleaner = TextCleaner()
    texts = ["Some text", "Some other text", "Yet another text"]
    result = cleaner.run(texts=texts)
    assert len(result) == 1
    assert result["texts"] == texts


def test_run_with_empty_inputs():
    cleaner = TextCleaner()
    result = cleaner.run(texts=[])
    assert len(result) == 1
    assert result["texts"] == []


def test_run_with_regex():
    cleaner = TextCleaner(remove_regexps=[r"\d+"])
    result = cleaner.run(texts=["Open123 Source", "HaystackAI"])
    assert len(result) == 1
    assert result["texts"] == ["Open Source", "HaystackAI"]


def test_run_with_multiple_regexps():
    cleaner = TextCleaner(remove_regexps=[r"\d+", r"[^\w\s]"])
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open Source", "HaystackAI"]


def test_run_with_convert_to_lowercase():
    cleaner = TextCleaner(convert_to_lowercase=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["open123! source", "haystack.ai"]


def test_run_with_remove_punctuation():
    cleaner = TextCleaner(remove_punctuation=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open123 Source", "HaystackAI"]


def test_run_with_remove_numbers():
    cleaner = TextCleaner(remove_numbers=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open! Source", "Haystack.AI"]


def test_run_with_multiple_parameters():
    cleaner = TextCleaner(
        remove_regexps=[r"\d+", r"[^\w\s]"], convert_to_lowercase=True, remove_punctuation=True, remove_numbers=True
    )
    result = cleaner.run(texts=["Open%123. !$Source", "Haystack.AI##"])
    assert len(result) == 1
    assert result["texts"] == ["open source", "haystackai"]
