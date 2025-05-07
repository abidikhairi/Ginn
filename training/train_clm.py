# train_clm.py

import sys
import logging
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from transformers.training_args import (
    EvaluationStrategy,
    OptimizerNames,
    SaveStrategy,
    SchedulerType,
    is_torch_bf16_gpu_available,
)
from peft import PeftConfig, PeftModel


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ScriptArguments:
    run_name: str = field(default="SmolLM2-1.7B-Instruct-CPPT")
    lora_model_path: str = field(default="./outputs/mlm")  # path to LoRA-adapted model
    dataset_path: str = field(default="data/sequences.csv")
    text_column: str = field(default="sequence")
    output_dir: str = field(default="./outputs/clm")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=1)
    lr: float = field(default=3e-4)

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    ############################
    ## Load model & tokenizer ##
    ############################
    tokenizer = AutoTokenizer.from_pretrained(args.lora_model_path)
    config = PeftConfig.from_pretrained(args.lora_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.lora_model_path)
    model = PeftModel(model, config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        def format_input(sequence: str) -> str:
            sequence = sequence.upper()
            sequence = " ".join(list(sequence))
            return f'<protein> {sequence} </protein>'
        
        sequences = [format_input(s) for s in examples[args.text_column]]
        
        return tokenizer(
            sequences,
            padding=True,
            truncation=True,
        )

    ###############
    ## Load Data ##
    ###############
    train_dataset = Dataset.from_csv(f'{args.dataset_path}/train.csv').select(range(10000))
    eval_dataset = Dataset.from_csv(f'{args.dataset_path}/validation.csv').select(range(5000))
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy=EvaluationStrategy.STEPS,
        save_strategy=SaveStrategy.STEPS,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        report_to="none",
        bf16=is_torch_bf16_gpu_available(),
        remove_unused_columns=True,
        optim=OptimizerNames.ADAMW_TORCH,
        lr_scheduler_type=SchedulerType.COSINE,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        run_name=args.run_name,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
