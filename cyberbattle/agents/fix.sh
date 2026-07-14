for f in \
  /cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/requirements.txt \
  /cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/setup.py \
  /cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/environment.yml; do
  sed -i \
    -e 's/nvidia-cublas-cu12==.*/nvidia-cublas-cu12>=12.1.3.1/' \
    -e 's/nvidia-cuda-cupti-cu12==.*/nvidia-cuda-cupti-cu12>=12.1.105/' \
    -e 's/nvidia-cuda-nvrtc-cu12==.*/nvidia-cuda-nvrtc-cu12>=12.1.105/' \
    -e 's/nvidia-cuda-runtime-cu12==.*/nvidia-cuda-runtime-cu12>=12.1.105/' \
    -e 's/nvidia-cudnn-cu12==.*/nvidia-cudnn-cu12>=9.1.0.70/' \
    -e 's/nvidia-cufft-cu12==.*/nvidia-cufft-cu12>=11.0.2.54/' \
    -e 's/nvidia-curand-cu12==.*/nvidia-curand-cu12>=10.3.2.106/' \
    -e 's/nvidia-cusolver-cu12==.*/nvidia-cusolver-cu12>=11.4.5.107/' \
    -e 's/nvidia-cusparse-cu12==.*/nvidia-cusparse-cu12>=12.1.0.106/' \
    -e 's/nvidia-nccl-cu12==.*/nvidia-nccl-cu12>=2.20.5/' \
    -e 's/nvidia-nvjitlink-cu12==.*/nvidia-nvjitlink-cu12>=12.6.68/' \
    -e 's/nvidia-nvtx-cu12==.*/nvidia-nvtx-cu12>=12.1.105/' \
    -e 's/sympy==.*/sympy>=1.13.1/' \
    -e 's/triton==.*/triton>=3.0.0/' \
    "$f" 2>/dev/null
done
