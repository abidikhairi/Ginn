import os
from argparse import ArgumentParser
from datasets import load_dataset


def main(args):
    dataset_id = 'khairi/swissprot-instructions'
    save_dir = args.save_dir
    
    dataset = load_dataset(dataset_id)
    
    splits = ('train', 'validation', 'test')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    for split in splits:
        dataset[split] = dataset[split].to_csv(f'{save_dir}/{split}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--save-dir', required=True, help='Save directory')
    
    args = parser.parse_args()
    main(args)
