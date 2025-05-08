from argparse import ArgumentParser
import logging
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import sys


MODEL_ID = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
MODEL_NAME = MODEL_ID.split('/')[-1]

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - SomlLM2 Loader - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args):
    model_path = args.model_path
    
    logger.info(f"Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = LlamaForCausalLM.from_pretrained(MODEL_ID)

    target_modules = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj', # self attention layer
        'gate_proj', 'up_proj', 'down_proj', # mlp layer
        'lm_head', 'tokens_embed' # tokens embedding layer
    ]

    logger.info(f"Preparing LoRA config with rank={args.rank}, alpha={args.alpha}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        target_modules=target_modules,
        lora_alpha=8,
        bias='none',   
    )

    logger.info("Preparing model with adapters")
    adapter = get_peft_model(model, peft_config, adapter_name='default')    

    adapter.print_trainable_parameters()

    logger.info(f"Saving models to {model_path}")    
    tokenizer.save_pretrained(f'{model_path}/base/{MODEL_NAME}')
    model.save_pretrained(f'{model_path}/base/{MODEL_NAME}', safe_serialization=False)
    adapter.save_pretrained(f'{model_path}/adapter/{MODEL_NAME}', safe_serialization=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--model-path', required=True, help='Model save path.')
    parser.add_argument('--rank', required=False, type=int, default=64, help='LoRA rank. Defaults 64.')
    parser.add_argument('--alpha', required=False, type=int, default=16, help='LoRA alpha. Defaults 8.')
    
    args = parser.parse_args()
    
    main(args)