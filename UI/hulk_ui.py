import sys
import argparse
import os
import time
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
print(trt.__version__)
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog,QLineEdit, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from pyModbusTCP.client import ModbusClient
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)


time_str = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(description=" ")
parser.add_argument('--command', default='run', help='')
parser.add_argument('--onnx', default='model.onnx', help='.')
parser.add_argument('--quant', default='fp32', help='')
parser.add_argument('--savepth', default='model.trt', help='')
parser.add_argument('--mdpth', default='model_10_16.trt', type=str)
# parser.add_argument('--impth', default='hulk_images', help='')
# parser.add_argument('--outpth', default='./res.png', help='')
args = parser.parse_args()

np.random.seed(123)
in_datatype = trt.nptype(trt.float32)
out_datatype = trt.nptype(trt.int32)
# ctx = pycuda.autoinit.context
trt.init_libnvinfer_plugins(None, "")
TRT_LOGGER = trt.Logger()


# 创建 Modbus TCP 客户端
SERVER_HOST = '192.168.1.5'  # Modbus TCP 设备的 IP 地址
SERVER_PORT = 502  # Modbus TCP 设备的端口号
client = ModbusClient(host=SERVER_HOST, port=SERVER_PORT)
client.open()
recording = False


def get_image(impath, size):
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
    var = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
    iH, iW = size[0], size[1]
    im = cv2.imread(impath)
    if iH == 256:
        im = im[0:256,:]
    img = im[:, :, ::-1]
    img = cv2.resize(img, (iW, iH)).astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = (img - mean) / var
    return img

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=in_datatype)
    d_input = cuda.mem_alloc(h_input.nbytes)
    h_outputs, d_outputs = [], []
    n_outs = 1
    for i in range(n_outs):
        h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(i + 1)), dtype=out_datatype)
        d_output = cuda.mem_alloc(h_output.nbytes)
        h_outputs.append(h_output)
        d_outputs.append(d_output)
    stream = cuda.Stream()
    return stream, h_input, d_input, h_outputs, d_outputs

def build_engine_from_onnx(onnx_file_path):
    engine = None
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
        assert os.path.exists(onnx_file_path), f'cannot find {onnx_file_path}'
        with open(onnx_file_path, 'rb') as fr:
            if not parser.parse(fr.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                assert False
        builder.max_batch_size = 128
        config.max_workspace_size = 1 << 30
        if args.quant == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
    return engine

def serialize_engine_to_file(engine, savepth):
    plan = engine.serialize()
    with open(savepth, "wb") as fw:
        fw.write(plan)

def deserialize_engine_from_file(savepth):
    with open(savepth, 'rb') as fr, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(fr.read())
    return engine

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('TensorRT Inference')
        self.setGeometry(100, 100, 800, 600)
        # self.resize(800,1000)


        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.save_path = ''

        self.compile_button = QPushButton('Compile Model', self)
        self.compile_button.clicked.connect(self.compile_model)

        self.run_button = QPushButton('Run Inference', self)
        self.run_button.clicked.connect(self.run_inference)

        self.save_button = QPushButton('Save Image', self)
        self.save_button.clicked.connect(self.save_image)

        self.select_save_path_button = QPushButton('Select Save Path', self)
        self.select_save_path_button.clicked.connect(self.select_save_path)

        self.run_plc_button = QPushButton('RUN PLC', self)
        self.run_plc_button.clicked.connect(self.run_plc)

        self.video_device_text = QLineEdit(self)
        self.video_device_text.setText('/dev/video3')
        self.review_restart_button = QPushButton('Restart Review', self)
        self.review_restart_button.clicked.connect(self.restart_review)
        hbox = QHBoxLayout()
        hbox.addWidget(self.video_device_text)
        hbox.addWidget(self.review_restart_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.compile_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.select_save_path_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.run_plc_button)
        layout.addLayout(hbox)


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = cv2.VideoCapture(self.video_device_text.text())
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        if not os.path.exists(args.onnx):
            print('ONNX file not found.')

            
        if not os.path.exists(args.mdpth):
            e = build_engine_from_onnx(args.onnx)
            serialize_engine_to_file(e, args.savepth)
            print('Model compiled and saved.')

        self.engine_r = deserialize_engine_from_file(args.mdpth)
        (
            stream,
            h_input,
            d_input,
            h_outputs,
            d_outputs,
        ) = allocate_buffers(self.engine_r)
        self.context = self.engine_r.create_execution_context()
        bds = [int(d_input), ] + [int(el) for el in d_outputs]
        dummy_data = np.random.rand(*self.engine_r.get_binding_shape(0)).astype(np.float32)
        cuda.memcpy_htod_async(d_input, dummy_data, stream)
        self.context.execute_async(bindings=bds, stream_handle=stream.handle)

        self.engine = None
        self.stream = stream
        self.h_input = h_input
        self.d_input = d_input
        self.h_outputs = h_outputs
        self.d_outputs = d_outputs
        self.OFFSET = 50
        self.plc = False

    def restart_review(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_device_text.text())

    def compile_model(self):
        self.engine = build_engine_from_onnx(args.onnx)
        serialize_engine_to_file(self.engine, args.savepth)
        print('Model compiled and saved.')

    def run_inference(self):
        ret, frame = self.cap.read()
        time1 = time.time()
        img = cv2.resize(frame, (self.engine_r.get_binding_shape(0)[2], self.engine_r.get_binding_shape(0)[1])).astype(np.float32)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = (img / 255.0 - 0.5) / 0.5
        img = np.ascontiguousarray(img)
        # np.copyto(self.h_input, img.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async(bindings=[int(self.d_input)] + [int(d) for d in self.d_outputs], stream_handle=self.stream.handle)
        for h_output, d_output in zip(self.h_outputs, self.d_outputs):
            cuda.memcpy_dtoh_async(h_output, d_output, self.stream)
        self.stream.synchronize()
        pred = self.h_outputs[0].reshape(self.engine_r.get_binding_shape(1))[0, :, :]
        print(f'Inference time: {time.time() - time1} seconds')
        pred = np.where(pred > 0, 255, 0).astype(np.uint8)
        row, col = np.nonzero(pred)
        max_col = max(col[row == 1]) 
        offset = self.OFFSET / 640 * max_col
        offset = round(offset, 2) * 100
        if self.plc:
            client.write_single_register(322, int(offset))

        # 下面是以轮廓面积来做做阈值判断
        # contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # max_area = 0
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area > max_area:
        #         max_area = area
        # # print('max_area:', max_area)  
        # if max_area > 27011 * 0.25:
        #     row, col = np.nonzero(pred)
        #     min_row = min(row)
        #     max_col = max(col[row == min_row])
        #     # min_col = min(col[row == min_row])
        #     # cv2.circle(origin_img, (min_col, min_row), 10, (0, 0, 255), -1)
        #     # cv2.circle(im_cp, (max_col, min_row), 10, (0, 255, 125), -1)
        #     # print('max_col:', max_col)
        #     offset = self.OFFSET / 640 * max_col
        #     offset = round(offset, 2) * 100
        #     # print('offset:', offset)
        #     # client.write_single_register(322, int(offset))
        #     # time2 = time.time()
        #     # print("postprecoseing Time cost vis is %lf seconds" % (time2 - time1))
        #     # with open('offsets.txt', 'a') as file:    
        #     #     file.write(f"{offset}\n")
        #     # time_str = time.strftime(f'%Y-%m-%d_%H-%M-%S_{i}', time.localtime(time.time()))
        #     # with open('offsets.txt', 'a') as file:
        #     #     file.write(f"{time_str}: {offset}\n")
            
        #     # pred = cv2.resize(pred, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        #     # pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        #     # dst = cv2.addWeighted(origin_img, 0.8, pred, 0.5, 0)
        #     # dst = cv2.circle(im_cp, (max_col, 1), 3, (0, 255, 0), -1)

        #     # time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        #     # out_result_path = 'out_result'
        #     # # os.makedirs(out_result_path, exist_ok=True)
        #     # cv2.imwrite(os.path.join(out_result_path, '1' + '.png'), dst)
        #     # cv2.imwrite(os.path.join(out_result_path, time_str + '.png'), dst)
        #     # i += 1

    def run_plc(self):
        ret, frame = self.cap.read()  # 先初始化相机
        while True:
            data = client.read_holding_registers(320, 1)[0]
            if data == 1001:
                print(data)
                self.plc = True
                self.run_inference()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            scaled_image = image.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
            # pix = QPixmap.fromImage(image)
            # self.video_label.setPixmap(pix)
        # if self.context:


        # img = QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        # pix = QPixmap.fromImage(img)
        # self.video_label.setPixmap(pix)

    def select_save_path(self):
        # self.save_path = QFileDialog.getSaveFileName(self, 'Select Save Path', '', 'Images (*.png *.xpm *.jpg)')[0]
        self.save_path = QFileDialog.getExistingDirectory(None, "Select Save Folder")
        print(self.save_path)
    def save_image(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        if self.save_path:
            t = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(self.save_path, f'{t}.png'), frame)
            print(f'Image saved to {self.save_path}')

    def closeEvent(self, event):
        # 关闭窗口时释放摄像头资源
        self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
