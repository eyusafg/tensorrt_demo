from __future__ import print_function

import os
import sys
import numpy as np
import tensorrt as trt
from PIL import ImageDraw
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
from data_pre import get_image
TRT_LOGGER = trt.Logger()
import cv2
import time


def get_engine(onnx_file_path, engine_file_path=""):
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            0
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 1 << 28
            )  # 256MiB
            
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(
                        onnx_file_path
                    )
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            network.get_input(0).shape = [1, 3, 480, 640]
            print("Completed parsing of ONNX file")
            print(
                "Building an engine from file {}; this may take a while...".format(
                    onnx_file_path
                )
            )
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():

    onnx_file_path = "model.onnx"
    engine_file_path = "test_trt.trt"
    img_pth = "hulk_images/0_.jpg"
    input_resolution_yolov3_HW = (480, 640)
    time1 = time.time()
    preprocessor = get_image(img_pth, input_resolution_yolov3_HW)

    # image_raw, image = preprocessor.process(input_image_path)
    
    # shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    output_shapes = [(1, 480, 640)]
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(
        onnx_file_path, engine_file_path
    ) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print("Running inference on image {}...".format(img_pth))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = preprocessor
        trt_outputs = common.do_inference(
            context,
            engine=engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [
        output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)
    ]
    print("Inference time: {:.3f}s".format(time.time() - time1))
    # Do post-processing
    pred = trt_outputs[0].reshape(input_resolution_yolov3_HW)
    pred = np.where(pred > 0, 255, 0).astype(np.uint8)
    # cv2.imshow('pred', pred)
    # cv2.waitKey(0)
    

if __name__ == "__main__":
    main()