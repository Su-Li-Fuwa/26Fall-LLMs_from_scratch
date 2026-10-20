from __future__ import annotations
import regex
#-------------------- P1.4: YOUR CODE HERE ----------------------
# class BPETokenizer:
#----------------------------------------------------------------
class BPETokenizer():
    
    def __init__(self, pattern: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.pattern = pattern
        self.vocab = vocab
        self.vocab_res = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_rank = {t:i for i,t in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        
        pretokens = regex.findall(self.pattern, text)
        tokens = []

        for pretoken in pretokens:
            encoding = [bytes([b]) for b in pretoken.encode('utf-8')]

            is_merged = 1
            while is_merged:
                idx, is_merged, to_merge = [], 0, []
                for idx in range(len(encoding)-1):
                    if (encoding[idx], encoding[idx+1]) in self.merges_rank:
                        if not is_merged : to_merge = [idx, self.merges_rank[(encoding[idx], encoding[idx+1])]]
                        elif self.merges_rank[(encoding[idx], encoding[idx+1])] < to_merge[-1]:
                            to_merge = [idx, self.merges_rank[(encoding[idx], encoding[idx+1])]]
                        is_merged = 1
                if is_merged:
                    encoding[to_merge[0]] += encoding[to_merge[0]+1]
                    encoding.pop(to_merge[0]+1)

            # for rule in self.merges:
            #     new_encoding, idx = [], 0
            #     while idx < len(encoding):
            #         if idx == len(encoding)-1:  
            #             new_encoding.append(encoding[idx])
            #             break
            #         if (encoding[idx], encoding[idx+1]) == rule:
            #             new_encoding.append(encoding[idx]+encoding[idx+1])
            #             idx += 2
            #         else:
            #             new_encoding.append(encoding[idx])
            #             idx += 1
            #     encoding = new_encoding

            for e in encoding:
                tokens.append(self.vocab_res[e])

        return tokens

    def decode(self, encoding: list[int]) -> str:
        if len(encoding) == 0:  return ''
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
    with open(corpus, "r", encoding="utf-8") as f:
        text = f.read()
    
    pretokens = regex.findall(pattern, text)
    encodings = [[bytes([b]) for b in pretoken.encode('utf-8')] for pretoken in pretokens]

    vocab = {b : bytes([b]) for b in range(256)}
    merges = []

    if len(vocab) >= vocab_size:    return vocab, merges

    pair = {}
    offset = [[] for _ in range(len(encodings))]

    def update_pair(p1, p2, pidx, idx):
        if (p1, p2) in pair:
            pair[(p1, p2)].append([pidx, idx])
        else:
            pair[(p1, p2)] = [[pidx, idx]]

    def remove_pair(p1, p2, pidx, idx):
        key = (p1, p2)

        if key not in pair:
            return

        pair[key].remove([pidx, idx])

        if not pair[key]:
            del pair[key]

    for pidx, pretoken in enumerate(encodings):
        for idx in range(len(pretoken)-1):
            update_pair(pretoken[idx], pretoken[idx+1], pidx, idx)
            

    def relative2absolute(off, idx):
        absolute_idx = idx
        for removed in off:
            if removed <= absolute_idx:
                absolute_idx += 1
            else:
                break
        return absolute_idx

    def absolute2relative(off, idx):
        k = 0
        for removed in off:
            if removed == idx:
                return None
            if removed < idx:
                k += 1
            else:
                break
        return idx - k

    while len(vocab) < vocab_size and pair:
        best_pair = max(pair, key=lambda p: (len(pair[p]), p))
        merges.append(best_pair)

        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        for pidx, idx in sorted(pair.pop(best_pair)):
            relative_idx = absolute2relative(offset[pidx], idx)

            if relative_idx is None:
                continue

            if relative_idx + 1 >= len(encodings[pidx]):
                continue

            if (
                encodings[pidx][relative_idx],
                encodings[pidx][relative_idx + 1]
            ) != best_pair:
                continue

            if relative_idx != 0:
                left_abs = relative2absolute(
                    offset[pidx], relative_idx - 1
                )

                remove_pair(
                    encodings[pidx][relative_idx - 1],
                    best_pair[0],
                    pidx,
                    left_abs
                )

                update_pair(
                    encodings[pidx][relative_idx - 1],
                    best_pair[0] + best_pair[1],
                    pidx,
                    left_abs
                )

            if relative_idx < len(encodings[pidx]) - 2:
                right_abs = relative2absolute(
                    offset[pidx], relative_idx + 1
                )

                remove_pair(
                    best_pair[1],
                    encodings[pidx][relative_idx + 2],
                    pidx,
                    right_abs
                )

                update_pair(
                    best_pair[0] + best_pair[1],
                    encodings[pidx][relative_idx + 2],
                    pidx,
                    idx
                )

            encodings[pidx][relative_idx] = (
                best_pair[0] + best_pair[1]
            )
            encodings[pidx].pop(relative_idx + 1)

            offset[pidx].append(idx + len(best_pair[0]))
            offset[pidx].sort()

    return vocab, merges

    #-----------------------------------------------------------------

# def train_bpe(
#     corpus: str,
#     vocab_size: int,
#     pattern: str,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Given the path to an input corpus, run train a BPE tokenizer and
#     output its vocabulary and merges.

#     Args:
#         corpus (str): Path to BPE tokenizer training data.
#         vocab_size (int): Total number of items in the tokenizer's vocabulary.
#         pattern (str): A regex pattern to pre-tokenize the input text.

#     Returns:
#         tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#             vocab:
#                 The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
#                 to bytes (token bytes)
#             merges:
#                 BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
#                 representing that <token1> was merged with <token2>.
#                 Merges are ordered by order of creation.
#     """

# #-------------------- P1.6a: YOUR CODE HERE ----------------------
#     with open(corpus, "r", encoding="utf-8") as f:
#         text = f.read()
#     pretokens = regex.findall(pattern, text)
#     encodings = [[bytes([b]) for b in pretoken.encode('utf-8')] for pretoken in pretokens]

#     vocab = {b : bytes([b]) for b in range(256)}
#     merges = []

#     if len(vocab) >= vocab_size:    return vocab, merges

#     pair = {}
#     offset = [[] for _ in range(len(encodings))]

#     def update_pair(pair, p1, p2, pidx, idx):

#         if (p1, p2) in pair:
#             pair[(p1, p2)].append([pidx, idx])
#         else:
#             pair[(p1, p2)] = [[pidx, idx]]

#         return pair 

#     for pidx, pretoken in enumerate(encodings):
#         for idx in range(len(pretoken)-1):
#             pair = update_pair(pair, pretoken[idx], pretoken[idx+1], pidx, idx)

#     def relative2absolute(off, idx):
#         for _ in range(len(off)):
#             if off[_] - _ - 1 > idx:
#                 return idx + _

#     def absolute2relative(off, idx):
#         k = 0
#         for _ in off: 
#             if _ < idx: k += 1
#         return idx - k

#     while len(vocab) < vocab_size:
#         best_pair = max(pair, key=lambda p: (len(pair[p]), p))
#         merges.append(best_pair)
#         vocab[len(vocab)] = best_pair[0] + best_pair[1]

#         for pidx, idx in pair[best_pair]:
#             relative_idx = absolute2relative(offset[pidx], idx)

#             if relative_idx != 0:
#                 pair[(encodings[pidx][relative_idx-1], best_pair[0])].remove([pidx, relative2absolute(offset[pidx],relative_idx-1)])
#                 pair = update_pair(pair, encodings[pidx][relative_idx-1], best_pair[0]+best_pair[1], pidx, relative2absolute(offset[pidx], relative_idx-1))

#             if relative_idx < len(encodings[pidx])-2:
#                 pair[(best_pair[1], encodings[pidx][relative_idx+2])].remove([pidx, relative2absolute(offset[pidx],relative_idx+2)])
#                 pair = update_pair(pair, best_pair[0]+best_pair[1], encodings[pidx][relative_idx+2], pidx, relative2absolute(offset[pidx],relative_idx+2))

#             encodings[pidx][relative_idx] = best_pair[0] + best_pair[1]
#             encodings[pidx][relative_idx].pop(relative_idx+1)
#             offset[pidx].append(idx + len(best_pair[0]))
#             offset[pidx].sort()

#     return vocab, merges