# python3 run.py --checkpoint facebook/opt-125m --compute-device cuda:0
# python3 run.py --checkpoint facebook/opt-125m --compute-device cuda:0 --normal-loop

# python3 run.py --checkpoint facebook/opt-13b --compute-device cuda:0
# python3 run.py --checkpoint facebook/opt-13b --compute-device cuda:0 --normal-loop

python3 run.py --checkpoint facebook/opt-2.7b --compute-device cuda:0
python3 run.py --checkpoint facebook/opt-2.7b --compute-device cuda:0 --normal-loop

nsys profile -o a.qdrep --force-overwrite true python3 run.py 

python3 run.py --checkpoint NousResearch/Llama-2-7b-chat-hf --compute-device cuda:0
python3 run.py --checkpoint huggyllama/llama-7b --compute-device cuda:0


python3 run.py --checkpoint facebook/opt-13b --compute-device cuda:0 --normal-loop --prompt-len 128 --gen-len 32 --gpu-batch-size 16 --num-gpu-batches 10 --percent 20 30 0 25 100 0 