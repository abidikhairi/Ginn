from transformers import AutoModelForCausalLM

def main():
    repo_id = 'khairi/SmolLM2-135M-Instruct'
    ckpt_path = "/ddn/data/fcit/jghazialharbi/experim/CP-Protein-LLM/checkpoint-1500"
    
    model = AutoModelForCausalLM.from_pretrained(ckpt_path)
    
    model.push_to_hub(repo_id)
    

if __name__ == "__main__":
    main()
