import os
from argparse import ArgumentParser
from datasets import load_dataset


def main(args):
    dataset_id = 'khairi/uniref50'
    save_dir = args.save_dir
    min_seq_len = args.min_seq_len
    max_seq_len = args.max_seq_len
    
    dataset = load_dataset(dataset_id)

    dataset = dataset.filter(lambda example: len(example['sequence']) >= min_seq_len and len(example['sequence']) <= max_seq_len)
    
    splits = ('train', 'validation', 'test')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    for split in splits:
        dataset[split] = dataset[split].to_csv(f'{save_dir}/{split}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--save-dir', required=True, help='Save directory')
    parser.add_argument('--min-seq-len', required=False, type=int, default=20, help='Minimum sequence length. Defaults 20')
    parser.add_argument('--max-seq-len', required=False, type=int, default=256, help='Maximum sequence length. Defaults 256')
    
    args = parser.parse_args()
    main(args)
