# from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.preprocessing import STOPWORDS
from gensim import utils
import os
import sys

try:
    from gensim.models.word2vec_inner import (  # noqa: F401
        train_batch_sg,
        train_batch_cbow,
        score_sentence_sg,
        score_sentence_cbow,
        MAX_WORDS_IN_BATCH,
        FAST_VERSION,
    )
except ImportError:
    raise utils.NO_CYTHON

if sys.version_info[0] >= 3:
    unicode = str

# changed default errors to 'ignore'
def any2unicode(text, encoding='utf8', errors='ignore'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.

    Returns
    -------
    str
        Unicode version of `text`.

    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)

to_unicode = any2unicode

class Text8Corpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip."""
    def __init__(self, fname, max_sentence_length=MAX_WORDS_IN_BATCH):
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest = [], b''
        with utils.open(self.fname, 'rb') as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    words = utils.to_unicode(text).split()
                    sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (to_unicode(text[:last_token]).split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]


def BuildPhraser(save_to_file=True,model_file_name=os.getcwd() + "/models/" + "bigram_model.pkl",
                 min_count=10, threshold=.7, common_terms=STOPWORDS,training_data=None):
    # Load training data.
    sentences = Text8Corpus(training_data)

    # Train bigram model.
    phrases = Phrases(sentences, min_count=min_count, threshold=threshold, common_terms=common_terms)

    # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    bigram_model = Phraser(phrases)

    # save the model to file
    if save_to_file:
        bigram_model.save(fname_or_handle=model_file_name)
    return bigram_model
