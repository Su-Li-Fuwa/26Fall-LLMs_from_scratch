from __future__ import annotations

#-------------------- P1.4: YOUR CODE HERE ----------------------
# class BPETokenizer:
#----------------------------------------------------------------
class BPETokenizer():
    
    def __init__(self, pattern: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.pattern = pattern
        self.vocab = vocab
        self.merges = merges

    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, encoding: list[int]) -> str:
        msg = self.vocab[encoding[0]]
        for v in encoding[1:]:
            msg += self.vocab[v]
        return msg.decode()
            
def train_bpe(
    corpus: str,
    vocab_size: int,
    pattern: str,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        corpus (str): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary.
        pattern (str): A regex pattern to pre-tokenize the input text.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    #-------------------- P1.6a: YOUR CODE HERE ----------------------
    raise NotImplementedError
    #-----------------------------------------------------------------
