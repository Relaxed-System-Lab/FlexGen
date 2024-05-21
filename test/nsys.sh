sudo sysctl -w kernel.perf_event_paranoid=2
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true -x true python cuda_stream.py

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report2 -f true -x true python cuda_stream.py


nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report4 -f true -x true python cuda_stream.py

x="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report5 -f true -x true python cuda_stream.py"
y=$(eval "$x")
nocache "$y"


nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --stats=true -s cpu -o nsight_report -f true -x true python profile_bandwidth.py

MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1  nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --stats=true -s cpu -o nvtx_cpu -f true -x true python nvtx_cpu.py


# --sample --cpuctxsw --event-sample --backtrace --cpu-core-events --event-sampling-frequency --os-events --samples-per-backtrace --sampling-period

ps -ef | grep dingfangyu | grep python | grep -v grep | awk '{print $2}' | xargs kill -9

sudo hdparm -tT /dev/nvme0n1p2
sudo hdparm -W /dev/nvme0n1p2
sudo hdparm -W 0 /dev/nvme0n1p2
sudo hdparm -W 1 /dev/nvme0n1p2

free -h

sudo sh -c 'echo 3 >  /proc/sys/vm/drop_caches'