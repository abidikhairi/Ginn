# train_sft.py

import sys
import logging
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)
from transformers.training_args import (
    OptimizerNames,
    IntervalStrategy,
    SchedulerType,
    is_torch_bf16_gpu_available,
)
from peft import PeftModel
from trl import SFTConfig, SFTTrainer

from callbacks import ModelInferenceCallback 


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - SFT - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ScriptArguments:
    run_name: str = field(default="SmolLM2-1.7B-Instruct-SwissProt")
    base_model_path: str = field(default="./_local/outputs/base-mlm")  # path to Base model
    lora_adapter_path: str = field(default="./_local/outputs/lora-mlm")  # path to LoRA-adapted model
    dataset_path: str = field(default="./_local/data")
    output_dir: str = field(default="./_local/outputs/sft")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: int = field(default=1)
    lr: float = field(default=2e-5)
    evaluation_steps: int = field(default=500)
    logging_steps: int = field(default=100)
    ckpt_output_dir: str = field(default='./_local/')


def main():
    parser = HfArgumentParser(ScriptArguments)
    args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    
    ############################
    ## Load model & tokenizer ##
    ############################
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
    model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)
    
    model.print_trainable_parameters()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def map_row_to_instruction(examples):
        conversations = []
        
        for i, s, o in zip(examples['input'], examples['system'], examples['output']):
            chat = [
                {'role': 'system', 'content': s},
                {'role': 'user', 'content': i},
                {'role': 'assistant', 'content': o},
            ]
            
            conversations.append(chat)
        
        return {'messages': conversations, }

    ###############
    ## Load Data ##
    ###############
    train_dataset = Dataset.from_csv(f'{args.dataset_path}/train.csv').select(range(128))
    eval_dataset = Dataset.from_csv(f'{args.dataset_path}/validation.csv').select(range(32))
    test_samples = Dataset.from_csv(f'{args.dataset_path}/test.csv').select(range(50, 53)).to_dict()
    
    train_dataset = train_dataset.map(map_row_to_instruction, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(map_row_to_instruction, batched=True, remove_columns=eval_dataset.column_names)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    training_args = SFTConfig(
        use_liger=True,
        dataset_num_proc=8,
        max_seq_length=512,
        packing=True,
        completion_only_loss=True,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        logging_steps=args.logging_steps,
        save_steps=args.evaluation_steps,
        eval_steps=args.evaluation_steps,
        report_to="none",
        bf16=is_torch_bf16_gpu_available(),
        remove_unused_columns=True,
        optim=OptimizerNames.ADAMW_TORCH,
        lr_scheduler_type=SchedulerType.COSINE,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        run_name=args.run_name,
        save_total_limit=3
    )
    
    model_inference_callback = ModelInferenceCallback(tokenizer=tokenizer, samples=test_samples, output_dir=args.ckpt_output_dir)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[model_inference_callback]
    )
    
    train_stats = trainer.train()
    
    print(train_stats)
        
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
