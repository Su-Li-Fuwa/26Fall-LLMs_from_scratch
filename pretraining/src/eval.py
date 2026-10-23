import argparse
import json
import re
from pathlib import Path

import torch


def find_files(root, task, specified=None):
    if specified:
        p = Path(specified)
        if not p.is_file():
            raise FileNotFoundError(p)
        files = [p]
    else:
        tags = {'hellaswag': ['hellaswag'], 'lambada': ['lambada'],
                'arc_easy': ['arc-easy', 'arc_easy', 'arc_e', 'arc-e']}[task]
        files = []
        for p in Path(root).rglob('*'):
            name = str(p).lower()
            if p.suffix not in ('.jsonl', '.json', '.parquet'):
                continue
            if any(tag in p.name.lower() for tag in ('contamination', 'annotation', 'metadata')):
                continue
            if not any(tag in name for tag in tags) or 'train' in p.name.lower():
                continue
            if task == 'hellaswag' and 'test' in p.name.lower() and 'val' not in p.name.lower():
                continue  # HellaSwag official test split has no labels
            if task == 'arc_easy' and ('challenge' in name or 'arc_c' in name):
                continue
            files.append(p)
        if not files:
            raise FileNotFoundError(f'Cannot find {task} in {root}; pass --{task} FILE')
        wanted = 'val' if task == 'hellaswag' else 'test'
        files.sort(key=lambda p: (wanted in p.name.lower(), p.suffix == '.jsonl'), reverse=True)
        files = [files[0]]
    # Include all shards if the selected Parquet split has multiple files.
    p = files[0]
    if p.suffix == '.parquet':
        prefix = re.sub(r'[-_]\d+(?:-of-\d+)?$', '', p.stem)
        files = sorted(p.parent.glob(prefix + '*.parquet'))
    return files


def read_rows(files):
    for file in files:
        if file.suffix == '.jsonl':
            with file.open(encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        elif file.suffix == '.json':
            rows = json.loads(file.read_text(encoding='utf-8'))
            if isinstance(rows, dict):
                rows = rows.get('data', rows.get('examples', rows.get('items', [rows])))
            yield from rows
        elif file.suffix == '.parquet':
            import pyarrow.parquet as pq  # pip install pyarrow
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                yield from batch.to_pylist()


def choices_of(raw):
    if isinstance(raw, dict):
        if 'text' in raw:
            return list(map(str, raw['label'])), list(map(str, raw['text']))
        return list(raw), list(map(str, raw.values()))
    if raw and isinstance(raw[0], dict):
        return ([str(x['label']) for x in raw], [str(x['text']) for x in raw])
    return list(map(str, range(len(raw)))), list(map(str, raw))


def example(task, x):
    if task == 'lambada':
        if 'prompt' in x and 'label' in x:
            return x['prompt'].rstrip(), [' ' + str(x['label']).strip()], 0

    if task == 'lambada':
        if 'target' in x or 'last_word' in x:
            word = str(x.get('target', x.get('last_word'))).strip()
            context = str(x.get('context', x.get('prompt', x.get('text', '')))).rstrip()
            if context.endswith(word):
                context = context[:-len(word)].rstrip()
        else:
            context, word = x['text'].rstrip().rsplit(maxsplit=1)
        return context, [' ' + word], 0

    if task == 'hellaswag':
        if 'query' in x:
            context, raw = x['query'], x['choices']
            gold = x.get('gold', x.get('label'))
        else:
            context = (f"{x['activity_label']}: {x['ctx_a']} {x['ctx_b'].capitalize()}"
                       if all(k in x for k in ('activity_label', 'ctx_a', 'ctx_b'))
                       else x.get('ctx', x.get('context', '')))
            raw, gold = x['endings'], x.get('label')
        _, ends = choices_of(raw)
        clean = lambda s: re.sub(r'\[.*?\]', '', s.replace(' [title]', '. ')).strip()
        return clean(context), [' ' + clean(s) for s in ends], int(gold)

    # ARC-Easy: supports official AI2 format and OpenCompass format.
    question = x.get('question', x.get('query', ''))
    raw = x.get('choices')

    if isinstance(question, dict):
        raw = raw if raw is not None else question.get('choices')
        question = question.get('stem', question.get('text', ''))

    if raw is not None:
        labels, ends = choices_of(raw)
    elif all('text' + c in x for c in 'ABCD'):
        labels = list('ABCD')
        ends = [x['text' + c] for c in 'ABCD']
    elif all(c in x for c in 'ABCD'):
        labels = list('ABCD')
        ends = [x[c] for c in 'ABCD']
    else:
        raise ValueError(f'Unknown ARC format: {list(x.keys())}')
    answer = str(x.get('answerKey', x.get('gold', x.get('answer'))))
    if answer in labels:
        gold = labels.index(answer)  # important for labels '1', '2', '3', '4'
    elif answer.upper() in 'ABCDE' and len(answer) == 1:
        gold = ord(answer.upper()) - ord('A')
    else:
        gold = int(answer)
    return str(question).strip(), [' ' + s.strip() for s in ends], gold


@torch.inference_mode()
def score(model, tok, context, endings, device, max_length=1024):
    """Return (sum log p, target token count, all-greedy-correct) for each ending."""
    items = []
    for ending in endings:
        prefix = tok.encode(context, add_special_tokens=False)
        target = tok.encode(ending, add_special_tokens=False)
        if not target or len(target) >= max_length:
            raise ValueError('Empty/overlong target; increase --max_length')
        if not prefix:
            prefix = [tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id]
        prefix = prefix[-(max_length - len(target)):]
        items.append((prefix + target, len(prefix), target))

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    length = max(len(ids) for ids, _, _ in items)
    ids = torch.full((len(items), length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, (tokens, _, _) in enumerate(items):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        mask[i, :len(tokens)] = 1

    try:
        out = model(input_ids=ids, attention_mask=mask)
    except TypeError as e:
        if 'attention_mask' not in str(e):
            raise
        out = model(input_ids=ids)  # for custom models without attention_mask
    logits = out if torch.is_tensor(out) else (out.logits if hasattr(out, 'logits') else out[0])

    results = []
    for i, (_, start, target) in enumerate(items):
        pred = logits[i, start - 1:start - 1 + len(target)].float()
        if not torch.isfinite(pred).all().item():
            raise ValueError(f'Non-finite logits for candidate {i}; check model weights and RoPE buffers')
        truth = torch.tensor(target, device=device)
        logp = pred.gather(1, truth[:, None]).flatten() - pred.logsumexp(-1)
        if not torch.isfinite(logp).all().item():
            raise ValueError(f'Non-finite log probabilities for candidate {i}')
        greedy_ok = bool((pred.argmax(-1) == truth).all().item())
        results.append((logp.sum().item(), len(target), greedy_ok))
    return results


def evaluate(task, files, model, tok, device, limit, max_length):
    correct = correct_norm = n = 0
    mean_logp = 0.0
    for row in read_rows(files):
        if limit and n >= limit:
            break
        context, endings, gold = example(task, row)
        if not 0 <= gold < len(endings):
            raise ValueError(f'Invalid gold label {gold} for {len(endings)} choices')
        scores = score(model, tok, context, endings, device, max_length)
        if task == 'lambada':
            correct += scores[0][2]  # every target sub-token must match argmax
            mean_logp += scores[0][0] / scores[0][1]
        else:
            correct += max(range(len(scores)), key=lambda i: scores[i][0]) == gold
            # Match lm-evaluation-harness: normalize by choice characters,
            # excluding the separator space inserted by example().
            lengths = [len(ending[1:]) for ending in endings]
            if not all(lengths):
                raise ValueError('Empty answer choice')
            correct_norm += max(range(len(scores)),
                                key=lambda i: scores[i][0] / lengths[i]) == gold
        n += 1
        if n % 500 == 0:
            print(f'  {task}: {n} samples', flush=True)
    if n == 0:
        raise ValueError(f'No samples in {files}')
    if task == 'lambada':
        return {'n': n, 'accuracy': correct / n, 'mean_subtoken_logp': mean_logp / n}
    return {'n': n, 'acc': correct / n, 'acc_norm': correct_norm / n}


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', default=str(root / 'exports/ckpt_18840'))
    p.add_argument('--tokenizer_dir', default=str(root / 'qwen3_tokenizer'))
    p.add_argument('--data_dir', default=str(root / 'dataset/benchmarks/data'))
    for task in ('hellaswag', 'lambada', 'arc_easy'):
        p.add_argument('--' + task, default=None, help='Override dataset file path')
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--limit', type=int, default=0, help='0 = all samples')
    p.add_argument('--output', default='eval_results.json')
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    results = {}
    for task in ('hellaswag', 'lambada', 'arc_easy'):
        files = find_files(a.data_dir, task, getattr(a, task))
        print(f'\n{task}: {[str(f) for f in files]}', flush=True)
        results[task] = evaluate(task, files, model, tok, device, a.limit, a.max_length)
        print(results[task], flush=True)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nSaved to {a.output}')


if __name__ == '__main__':
    main()
