sudo sysctl -w kernel.perf_event_paranoid=2

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o rep/nsight_report -f true -x true python cuda_stream.py

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o rep/nsight_report2 -f true -x true python cuda_stream.py


nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o rep/nsight_report4 -f true -x true python cuda_stream.py

x="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o rep/nsight_report5 -f true -x true python cuda_stream.py"
nocache `nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o rep/nsight_report5 -f true -x true python cuda_stream.py | cat >>/dev/null`


nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --stats=true -s cpu -o rep/nsight_report -f true -x true python profile_bandwidth.py

MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1  nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --stats=true -s cpu -o nvtx_cpu -f true -x true python nvtx_cpu.py


ps -ef | grep dingfangyu | grep 'miniconda3/bin/python' | grep -v grep | awk '{print $2}' | xargs kill -9


sudo hdparm -tT --direct /dev/nvme0n1p2
sudo hdparm -tT /dev/nvme0n1p2
sudo hdparm -W /dev/nvme0n1p2
sudo hdparm -W 0 /dev/nvme0n1p2
sudo hdparm -W 1 /dev/nvme0n1p2

tune2fs -l /dev/nvme0n1p2 | grep -i 'block size' 

free -h

sudo sh -c 'echo 3 >  /proc/sys/vm/drop_caches'

sudo cp *.so /usr/local/lib/
sudo cp *.sh /usr/local/bin/

pagecache-management.sh nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o rep/nsight_report15 -f true -x true python cuda_stream.py 

find . -size +100k | sed 's|^\./||g' | cat > .gitignore && git add . && git commit -m 'chunk_size, chunk_dim, chunk_num' && git push