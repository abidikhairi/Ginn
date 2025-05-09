import json
import time
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback
)


class ModelInferenceCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        samples: List[Dict],
        output_dir: str
    ):
        super().__init__()
        self.tokenizer: AutoTokenizer = tokenizer
        self.samples: List[Dict] = samples
        self.output_dir = output_dir
        
    
    def run_inference(self, model: AutoModelForCausalLM):
        labels = []
        predictions = []
        prompts = []
        for s, i, label in zip(self.samples['system'], self.samples['input'], self.samples['output']):    
            inputs = [{'role': 'system', 'content': s}, {'role': 'user', 'content': i}]
            inputs = self.tokenizer.apply_chat_template(inputs, return_dict=True, add_generation_prompt=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=250, top_p=0.9)
            output = outputs[0]
            output = output[inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(output)

            prompts.append(i)
            labels.append(label)
            predictions.append(output)
            
        return labels, predictions, prompts

    def on_evaluate(self, args, state, control, **kwargs):
        model: AutoModelForCausalLM = kwargs.pop('model')
        filename = f'{self.output_dir}/{int(time.time())}.json'
        
        if state.is_local_process_zero and state.is_world_process_zero:
            labels, predictions, prompts = self.run_inference(model)
            data = [
                {'label': l, 'prediction': p, 'prompt': p1} for l, p, p1 in zip(labels, predictions, prompts)
            ]
            
            with open(filename, 'w') as f:
                json.dump(data, f)
                
        return super().on_evaluate(args, state, control, **kwargs)


class GenerateProteinsCallback(TrainerCallback):
    def __init__(self, tokenizer: AutoTokenizer, output_dir: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        
    def on_evaluate(self, args, state, control, **kwargs):
        model: AutoModelForCausalLM = kwargs.pop('model')
        input = '<protein>'
        sequences = []
        filename = f'{self.output_dir}/{int(time.time())}.txt'
        for _ in range(5):
            inputs = self.tokenizer(input, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=250, top_p=0.9)
            output = outputs[0]
            sequence:str = self.tokenizer.decode(output)
            sequence = sequence.split('</protein>')[0].replace('<protein>', '').strip()
            sequences.append(sequence)
            
            with open(filename, 'w') as f:
                for s in sequences:
                    f.write(f'{s}\n')
    
        return super().on_evaluate(args, state, control, **kwargs)