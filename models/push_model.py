# push_model.py
import argparse
from transformers import AutoModelForCausalLM
from peft import PeftConfig, PeftModel


def main(args):
    repo_id = args.repo_id
    ckpt_path = args.ckpt_path    
    
    if not args.push_only_adapter:
        model = AutoModelForCausalLM.from_pretrained(ckpt_path)
        model.push_to_hub(repo_id)
    else:
        config = PeftConfig.from_pretrained(ckpt_path)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model.push_to_hub(repo_id)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--repo-id', required=True, help="Model HF repository")
    parser.add_argument('--ckpt-path', required=True, help="Checkpoint path")
    parser.add_argument('--push-only-adapter', action='store_true', help="Push only the LoRA adapter")
        
    args = parser.parse_args()
    main(args)
