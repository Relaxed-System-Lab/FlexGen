sudo sysctl -w kernel.perf_event_paranoid=2
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true -x true python cuda_stream.py