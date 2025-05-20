import argparse
import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


def main(args):
    dataset_id = 'khairi/uniref50'
    save_dir = args.save_dir
    min_seq_len = args.min_seq_len
    max_seq_len = args.max_seq_len

    dataset = load_dataset(dataset_id)

    dataset = dataset.filter(lambda example: len(example['sequence']) >= min_seq_len and len(example['sequence']) <= max_seq_len)

    print(dataset)
    
    splits = ('train', 'validation', 'test')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for split in splits:
        
        if split == 'train':
            ds = dataset[split].shuffle(seed=1234).select(range(args.num_train_samples))
        else:
            ds = dataset[split].shuffle(seed=42)
        
        print(f"Converting {split} dataset to pandas dataframe")
        df = ds.to_pandas()
        
        print("Reversing sequences")
        df['reverse_sequence'] = df['sequence'].apply(lambda x: ''.join(reversed(list(x))))
        df1 = df[['entry', 'reverse_sequence']]
        df1.columns = ['entry', 'sequence']
        print("Reversing sequences done")
        
        df2 = df[['entry', 'sequence']]

        print("Concatenating original and reversed sequences")
        df = pd.concat([df1, df2], ignore_index=True)

        print("Saving sequences")
        df.to_csv(f'{save_dir}/{split}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample sequences from a dataset.")
    
    parser.add_argument('--save-dir', required=True, help='Save directory')
    parser.add_argument('--min-seq-len', type=int, default=20, help='Minimum sequence length. Defaults to 20')
    parser.add_argument('--max-seq-len', type=int, default=512, help='Maximum sequence length. Defaults to 256')
    parser.add_argument('--num-train-samples', type=int, default=2_000_000, help='Number of training sequences. Defaults to 2M')
    
    args = parser.parse_args()
    main(args)

# This script samples sequences from the khairi/uniref50 dataset and saves them to CSV files.
# example usage:
# python data/pretraining/sample_sequences.py --save-dir ./data/uniref50-sample01 --min-seq-len 20 --max-seq-len 512 --num-train-samples 2000000