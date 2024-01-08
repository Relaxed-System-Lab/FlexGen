python -m flexgen.flex_opt --model facebook/opt-125m --gpu-batch-size 32 --percent 10 20 100 0 100 0

python -m flexgen.flex_opt --model facebook/opt-13b --prompt-len 10 --gen-len 3 --gpu-batch-size 32 --num-gpu-batches 10 --percent 20 30 0 25 100 0 

nsys profile -o flexgen.qdrep --force-overwrite true python -c "import os; os.system(\"python -m flexgen.flex_opt --model facebook/opt-13b --prompt-len 10 --gen-len 3 --gpu-batch-size 32 --num-gpu-batches 10 --percent 20 30 0 25 100 0 \")"