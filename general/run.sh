# python3 run.py --checkpoint facebook/opt-125m --compute_device cuda:0
# python3 run.py --checkpoint facebook/opt-125m --compute_device cuda:0 --normal_loop

# python3 run.py --checkpoint facebook/opt-13b --compute_device cuda:0
# python3 run.py --checkpoint facebook/opt-13b --compute_device cuda:0 --normal_loop

python3 run.py --checkpoint facebook/opt-2.7b --compute_device cuda:0
python3 run.py --checkpoint facebook/opt-2.7b --compute_device cuda:0 --normal_loop

nsys profile -o a.qdrep --force-overwrite true python3 run.py 

python3 run.py --checkpoint NousResearch/Llama-2-7b-chat-hf --compute_device cuda:0
