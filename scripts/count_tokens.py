"""
Count total tokens in the dataset.
"""
import argparse
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description='Count tokens in dataset')
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split')
parser.add_argument('--max_docs', type=int, default=None, help='Maximum documents to process (default: all)')
parser.add_argument('--tokenizer_threads', type=int, default=8, help='Number of tokenizer threads')
args = parser.parse_args()

print(f"Counting tokens in {args.split} split...")
tokenizer = get_tokenizer("hansard")

total_tokens = 0
total_docs = 0
total_chars = 0

for batch in parquets_iter_batched(split=args.split):
    # Tokenize the batch
    token_lists = tokenizer.encode(batch, prepend=tokenizer.get_bos_token_id(), num_threads=args.tokenizer_threads)

    # Count tokens
    batch_tokens = sum(len(tokens) for tokens in token_lists)
    batch_chars = sum(len(doc) for doc in batch)

    total_tokens += batch_tokens
    total_docs += len(batch)
    total_chars += batch_chars

    if total_docs % 10000 == 0:
        print(f"Processed {total_docs:,} docs | {total_tokens:,} tokens | {total_chars:,} chars | {total_chars/total_tokens:.2f} chars/token")

    if args.max_docs and total_docs >= args.max_docs:
        break

print("\n" + "="*80)
print(f"Final counts for {args.split} split:")
print(f"  Total documents: {total_docs:,}")
print(f"  Total characters: {total_chars:,}")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Chars per token: {total_chars/total_tokens:.2f}")
print(f"  Tokens per document (avg): {total_tokens/total_docs:.1f}")
print("="*80)
