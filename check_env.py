# check_env.py
import torch
import transformers
import peft
import datasets
import accelerate


def main():
    print("Library Versions:")
    print(f"  torch        : {torch.__version__}")
    print(f"  transformers : {transformers.__version__}")
    print(f"  peft         : {peft.__version__}")
    print(f"  datasets     : {datasets.__version__}")
    print(f"  accelerate   : {accelerate.__version__}")
    
    print("-" * 40)
    num_devices = torch.cuda.device_count()
    
    print(f'Number of available devices: {num_devices} GPUs')
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1e9  # GB
        reserved = torch.cuda.memory_reserved(i) / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        free = reserved - allocated

        print(f"GPU {i} - {props.name}")
        print(f"  Total Memory   : {total:.2f} GB")
        print(f"  Reserved Memory: {reserved:.2f} GB")
        print(f"  Allocated Memory: {allocated:.2f} GB")
        print(f"  Free (within reserved): {free:.2f} GB\n")



if __name__ == '__main__':
    main()
