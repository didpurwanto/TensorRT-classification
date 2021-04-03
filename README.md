# TensorRT-classification
TensorRT for jetson Nano


Dependencies: <br />
PyTorch 1.2.0<br />
Pycuda <install from the scratch>
  https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pycuda<br />
CUDA 10.0<br />
Python3.6<br />
albumentations==0.4.5<br />
onnx==1.4.1<br />
opencv-python==4.2.0.34<br /><br />
  
  Download and install TensorRT https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html <br /><br />
  
Notes:
~/.bashrc
export LD_LIBRARY_PATH=/home/didpurwanto/Documents/important/TensorRT-6.0.1.5/lib:$LD_LIBRARY_PATH <br />
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-6.0.1.5/lib<br /><br /><br />


Thanks to: <br />
https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/

