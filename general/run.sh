# python3 run.py --checkpoint facebook/opt-125m --compute-device cuda:0
# python3 run.py --checkpoint facebook/opt-125m --compute-device cuda:0 --normal-loop

# python3 run.py --checkpoint facebook/opt-13b --compute-device cuda:0
# python3 run.py --checkpoint facebook/opt-13b --compute-device cuda:0 --normal-loop

python3 run.py --checkpoint facebook/opt-125m --compute-device cuda:0 --normal-loop --prompt-len 32 --gen-len 10 --gpu-batch-size 2 --num-gpu-batches 4 --percent 20 30 0 25 0 100  

python3 run.py --checkpoint facebook/opt-2.7b --compute-device cuda:0
python3 run.py --checkpoint facebook/opt-2.7b --compute-device cuda:0 --normal-loop

nsys profile -o a.qdrep --force-overwrite true python3 run.py 

python3 run.py --checkpoint NousResearch/Llama-2-7b-chat-hf --compute-device cuda:0
python3 run.py --checkpoint huggyllama/llama-7b --compute-device cuda:0

python3 run.py --checkpoint facebook/opt-13b --compute-device cuda:0 --normal-loop --prompt-len 128 --gen-len 32 --gpu-batch-size 16 --num-gpu-batches 10 --percent 20 30 0 25 100 0 
python3 run.py --checkpoint facebook/opt-13b --compute-device cuda:0 --normal-loop --prompt-len 128 --gen-len 32 --gpu-batch-size 8 --num-gpu-batches 10 --percent 20 30 0 25 0 100  

# watch GPU memory usage
watch -n 0.1 nvidia-smi

# requirements
pipreqs . --encoding utf-8 --force