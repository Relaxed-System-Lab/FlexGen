python -m flexgen.flex_opt --model facebook/opt-125m --gpu-batch-size 32 --percent 10 20 100 0 100 0

python -m flexgen.flex_opt --model facebook/opt-13b --prompt-len 10 --gen-len 3 --gpu-batch-size 32 --num-gpu-batches 10 --percent 20 30 0 25 100 0 

nsys profile -o flexgen.qdrep --force-overwrite true python -c \
 "import os; os.system(\"python -m flexgen.flex_opt \
 --model facebook/opt-13b --sep-layer False \
 --prompt-len 64 --gen-len 2 \
 --gpu-batch-size 32 --num-gpu-batches 4 \
 --percent 20 30 0 25 100 0 \")"

watch -n 0.1 nvidia-smi

find . -size +1M | sed 's|^\./||g' | cat > .gitignore 

find . -size +100k | sed 's|^\./||g' | cat > .gitignore && git add . && git commit -m 'pin memory' && git push