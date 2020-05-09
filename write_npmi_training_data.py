from __future__ import absolute_import, division, print_function
import json
import logging
from io import open
from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from cdqa.reader.bertqa_sklearn import SquadExample, _is_whitespace
logger = logging.getLogger(__name__)

class SquadExampleJB(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        doc_tokens,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=None,
        paragraph=None,
        title=None,
        retriever_score=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.paragraph = paragraph
        self.title = title
        self.retriever_score = retriever_score

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)

        s += ", orig_answer_text: [%s]" % (" ".join(self.orig_answer_text))
        s += ", paragraph: [%s]" % (" ".join(self.paragraph))
        return s



def read_squad_examples_jb(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""

    if isinstance(input_file, str):
        with open(input_file, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]
    else:
        input_data = input_file

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                try:
                    retriever_score = qa["retriever_score"]
                except KeyError:
                    retriever_score = 0
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer."
                        )
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[
                            answer_offset + answer_length - 1
                        ]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position : (end_position + 1)]
                        )
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text)
                        )
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning(
                                "Could not find answer: '%s' vs. '%s'",
                                actual_text,
                                cleaned_answer_text,
                            )
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                examples.append(
                    SquadExampleJB(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                        paragraph=paragraph_text,
                        title=entry["title"],
                        retriever_score=retriever_score,
                    )
                )
    return examples

'''
X='./data/SQuAD_1.1/train-v1.1.json'
examples = read_squad_examples_jb(
            input_file=X,
            # is_training=self.is_training,
            is_training=True,
            # version_2_with_negative=self.version_2_with_negative,
            version_2_with_negative=False
        )
'''
