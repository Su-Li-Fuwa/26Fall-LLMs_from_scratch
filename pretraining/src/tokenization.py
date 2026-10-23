from transformers import AutoTokenizer
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Tokenize local FineWeb-Edu Parquet files to int32 streams")
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--tokenizer_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()
    seed = 20261021
    rng = np.random.default_rng(seed=seed)
    valid_ratio = 2/1000 
    batch_size = 512
    num_workers = 16
    
    current_dir = Path(__file__).resolve().parent.parent
    
    path_data = Path(args.input_dir)
    parquet_files = sorted(path_data.glob("*.parquet"))
    
    path_qwenTokenizer = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_train, path_valid = output_dir / 'train.bin', output_dir / 'valid.bin'
    if not parquet_files:
        raise FileNotFoundError(f'No Parquet files in {path_data}')
    if path_train.exists() or path_valid.exists():
        raise FileExistsError('Choose an empty output directory to avoid overwriting token streams')
    
    tokenizer = AutoTokenizer.from_pretrained(
        path_qwenTokenizer,
        use_fast=True,
        local_files_only=True
    )
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    def tokenize(text):
        return tokenizer.encode(text, add_special_tokens=False) + [eot_id]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor, open(path_train, "wb") as f_train, open(path_valid, "wb") as f_valid:
    
        for path in parquet_files:
            pf = pq.ParquetFile(path)
            print(path)
            for batch in pf.iter_batches(batch_size = batch_size, columns=["text"]):
                train_l, valid_l = [], []
    
                texts = (text for text in batch.column(0).to_pylist() if text is not None)
    
                for token in executor.map(tokenize, texts):
                    if rng.random() < valid_ratio:
                        valid_l.extend(token)
                    else:
                        train_l.extend(token)
    
                np.array(train_l, dtype=np.int32).tofile(f_train)
                np.array(valid_l, dtype=np.int32).tofile(f_valid)
        

if __name__ == "__main__":
    main()
