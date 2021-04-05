Deployment on Server 1080ti

Reqs:
torch==1.2.0
torchvision==0.4.0
albumentations==0.4.5
onnx==1.4.1
opencv-python==4.2.0.34 
python3.6 
CUDA10.0


Install TensorRT
Download CUDA10.0 https://developer.nvidia.com/cuda-10.0-download-archive
Download CUDNN https://developer.nvidia.com/rdp/cudnn-download
Download TensorRT https://developer.nvidia.com/nvidia-tensorrt-6x-download

Step by step to install TensorRT https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
1. Unrar the TensorRT-7.0.0.11
2. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
e.g. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/didpurwanto/Documents/important/TensorRT-7.0.0.11/lib or put it in ~/.bashrc fild
3. cd python
sudo pip3 install tensorrt-*-cp3x-none-linux_x86_64.whl
4. cd uff
sudo pip3 install uff-0.6.9-py2.py3-none-any.whl
check with which convert-to-uff
5. cd graphsurgeon
sudo pip3 install graphsurgeon-0.4.5-py2.py3-none-any.whl
6. cd onnx_graphsurgeon
sudo pip3 install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl

If you want to upgrade from TensorRT6.0 to TensorRT7.0, you should force-reinstall it
e.g: sudo pip3 install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl --force-reinstall it

7. check the library with tree -d 


Install PyCUDA
1. Download pycuda-2019.1.1
https://pypi.org/project/pycuda/2019.1.1/#files
2. UnRAR pycuda-2019.1.1
3. cd pycuda-2019.1.1
4. python3.6 configure.py --cuda-root=/usr/local/cuda-10.0, check siteconf.py for the resulted config.
5. sudo su -c "make install" if failed, check your dependencies on python. 
I used Python3.6, but the dependencies are on Python3.5. Solution:
sudo CPLUS_INCLUDE_PATH=/usr/include/python3.5 make all
sudo CPLUS_INCLUDE_PATH=/usr/include/python3.5 make install
Then, install one more time.
sudo su -c "make install"

Check again
cd pycuda-2019.1.1/test
python3.6 test_driver.py
If it says "OK", it means that the installation of PyCUDA was successful.


Optional: Boost C++ Libraries
python3.6 configure.py \
  --boost-inc-dir=$HOME/pool/include/boost-1_35 \
  --boost-lib-dir=$HOME/pool/lib \
  --boost-python-libname=boost_python-gcc42-mt \
  --boost-thread-libname=boost_thread-gcc42-mt \
  --cuda-root=/usr/local/cuda-10.0
sudo make all 
sudo su -c "make install"




