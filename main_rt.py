import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch as t
import cv2
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import  ToTensor
from albumentations.augmentations.transforms import Normalize

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print('kkk')
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 1
    if builder.platform_has_fast_fp16:
    	builder.fp16_mode = True

    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context

def calculate_score(output_data):
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    
    confidences = t.nn.functional.softmax(output_data, dim=1)[0] * 100
    _, indices = t.sort(output_data, descending=True)
    
    # print('indices ', indices)
    # print('confidences: ', confidences.shape)
    # print('classes: ', classes)

    for i in  range(1000):
        score = confidences[i].cpu().detach().numpy()
        if score > 0.5:
            print(score, classes[i])


def preprocess_image(img_path):
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ])
    
    input_img = cv2.imread(img_path)

    input_data = transforms(image=input_img)["image"]
    batch_data = t.unsqueeze(input_data, 0)
    return batch_data

def main():
    print('a')
    engine, context = build_engine('resnet50.onnx')	
    print('start')
    for binding in engine:
	    if engine.binding_is_input(binding):  # we expect only one input
	        input_shape = engine.get_binding_shape(binding)
	        input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
	        device_input = cuda.mem_alloc(input_size)
	    else:  # and one output
	        output_shape = engine.get_binding_shape(binding)
	        # create page-locked memory buffers (i.e. won't be swapped to disk)
	        host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
	        device_output = cuda.mem_alloc(host_output.nbytes)
    stream = cuda.Stream()
    host_input = np.array(preprocess_image("/media/didpurwanto/DiskL/disertation_ex/turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    output_data = t.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    print('output_data', output_data)
    calculate_score(output_data)


main()    
