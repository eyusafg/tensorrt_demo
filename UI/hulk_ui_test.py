import sys
import argparse
import os
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
print(trt.__version__)
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog,QLineEdit, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pyModbusTCP.client import ModbusClient
from os.path import join, dirname, realpath
cur_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(cur_dir)
# import matplotlib.pyplot as plt
import lib.segmentation.data.transform_cv2 as T
from models.segmentation import model_factory
from configs.segmentation import set_cfg_from_file


time_str = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(description=" ")
parser.add_argument('--command', default='run', help='')
parser.add_argument('--onnx', default='model.onnx', help='.')
parser.add_argument('--quant', default='fp32', help='')
parser.add_argument('--savepth', default='model.trt', help='')
parser.add_argument('--mdpth', default='model_10_32.trt', type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--config', dest='config', type=str, default='configs/segmentation/bisenetv2_syt_segm_edge_hulk_0529.py',) 
parser.add_argument('--weight-path', type=str, default='model_10.pth',)  
parser.add_argument('--impth', default='hulk_images', help='')
# parser.add_argument('--outpth', default='./res.png', help='')
args = parser.parse_args()


torch.set_grad_enabled(False)
np.random.seed(123)
in_datatype = trt.nptype(trt.float32)
out_datatype = trt.nptype(trt.int32)
# ctx = pycuda.autoinit.context
trt.init_libnvinfer_plugins(None, "")
TRT_LOGGER = trt.Logger()



def get_image(im):
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
    var = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
    # iH, iW = size[0], size[1]
    # im = cv2.imread(im)
    # if iH == 256:
    #     im = im[0:256,:]
    img = im[:, :, ::-1].astype(np.float32)
    # img = cv2.resize(img, (iW, iH)).astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = (img - mean) / var
    # img -= mean
    # img /= var
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
        self.setGeometry(100, 100, 1200, 600)
        # self.resize(800,1000)


        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setGeometry(10, 10, 320, 480)
        self.save_path = ''

        self.cpu_label = QLabel(self)  
        self.cpu_label.setAlignment(Qt.AlignCenter)
        self.cpu_label.setText("CPU Info")
        
        self.cuda_label = QLabel(self)  
        self.cuda_label.setAlignment(Qt.AlignCenter)
        self.cuda_label.setText("CUDA Info")
        
        self.trt_label = QLabel(self) 
        self.trt_label.setAlignment(Qt.AlignCenter)
        self.trt_label.setText("TensorRT Info")


        self.compile_button = QPushButton('Compile Model', self)
        self.compile_button.clicked.connect(self.compile_model)

        self.run_button = QPushButton('Run Inference', self)
        self.run_button.clicked.connect(self.infer_all)

        self.save_button = QPushButton('Save Image', self)
        self.save_button.clicked.connect(self.save_image)

        self.select_save_path_button = QPushButton('Select Save Path', self)
        self.select_save_path_button.clicked.connect(self.select_save_path)

        # self.run_plc_button = QPushButton('RUN PLC', self)
        # self.run_plc_button.clicked.connect(self.run_plc)

        self.video_device_text = QLineEdit(self)
        self.video_device_text.setText('/dev/video3')
        self.review_restart_button = QPushButton('Restart Review', self)
        self.review_restart_button.clicked.connect(self.restart_review)

        hbox = QHBoxLayout()
        hbox.addWidget(self.video_device_text)
        hbox.addWidget(self.review_restart_button)

        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.video_label)
        hbox_labels.addWidget(self.cpu_label)
        hbox_labels.addWidget(self.cuda_label)
        hbox_labels.addWidget(self.trt_label)

        layout = QVBoxLayout()
        layout.addLayout(hbox_labels)
        # layout.addWidget(self.video_label)
        layout.addWidget(self.compile_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.select_save_path_button)
        layout.addWidget(self.save_button)
        # layout.addWidget(self.run_plc_button)
        layout.addLayout(hbox)
        # layout.addLayout(hbox_labels)


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

        self.cfg = set_cfg_from_file(args.config)
        self.cfg_dict = dict(self.cfg.__dict__)
        self.in_channel = self.cfg_dict['in_ch']

        self.net = model_factory[self.cfg.model_type](self.cfg.n_cats,in_ch=self.in_channel, aux_mode='eval', net_config=self.cfg.net_config)
        self.net1 = model_factory[self.cfg.model_type](self.cfg.n_cats,in_ch=self.in_channel, aux_mode='eval', net_config=self.cfg.net_config)
        self.net.load_state_dict(torch.load(args.weight_path, map_location='cuda'), strict=False)
        self.net1.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
        self.net.cuda()
        self.net.eval()
        self.net1.eval()

        target_size = self.cfg_dict['target_size']
        dummy_input = torch.randn(self.in_channel, target_size[0], target_size[1])
        # dummy_input = torch.randn(in_channel, 480, 640)
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        normalized_input = (dummy_input - mean[None, :, None, None]) / std[None, :, None, None]
        normalized_input = normalized_input.cuda()
        for i in range(10):
            _ = self.net(normalized_input)

    def restart_review(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_device_text.text())
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.showNormal()
        super().keyPressEvent(event)

    def pt_infer_cuda(self, frame):
        # ret, frame = self.cap.read()
        # cfg = set_cfg_from_file(args.config)
        # cfg_dict = dict(cfg.__dict__)
        # in_channel = cfg_dict['in_ch']

        # net = model_factory[cfg.model_type](cfg.n_cats,in_ch=in_channel, aux_mode='eval', net_config=cfg.net_config)
        # net.load_state_dict(torch.load(args.weight_path, map_location='cuda'), strict=False)
        # net.cuda()
        # net.eval()

        to_tensor = T.ToTensor(mean=(0.5, 0.5, 0.5),  std=(0.5, 0.5, 0.5),)
        im = frame[:, :, ::-1]
        # time1 = time.time()
        im = np.ascontiguousarray(im)
        im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)
        im = im.cuda()
        time1 = time.time()
        out = self.net(im)[0]
        out = out.argmax(dim=1)
        out = out.squeeze().detach().cpu().numpy()
        pred_cuda = np.where(out > 0, 255, 0).astype(np.uint8)
        time2 = time.time()
        coast_time = time2 - time1  
        print('pt_infer_cuda time cost is %lf seconds' % coast_time)
        # cv2.imshow('pred_cuda', pred_cuda)
        # cv2.waitKey(0)
        return pred_cuda, coast_time

    def pt_infer_cpu(self, frame):
        # ret, frame = self.cap.read()
        # cfg = set_cfg_from_file(args.config)
        # cfg_dict = dict(cfg.__dict__)
        # in_channel = cfg_dict['in_ch']

        # net = model_factory[cfg.model_type](cfg.n_cats,in_ch=in_channel, aux_mode='eval', net_config=cfg.net_config)
        # net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
        # net.eval()

        to_tensor = T.ToTensor(mean=(0.5, 0.5, 0.5),  std=(0.5, 0.5, 0.5),)
        # time1 = time.time()
        im = frame[:, :, ::-1]
        im = np.ascontiguousarray(im)
        im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)
        time1 = time.time()

        out = self.net1(im)[0]
        out = out.argmax(dim=1)
        # out = out.squeeze().detach().cpu().numpy()
        pred_cpu = np.where(out > 0, 255, 0).astype(np.uint8)
        pred_cpu = pred_cpu.transpose(1, 2, 0).astype(np.uint8)
        time2 = time.time()
        coast_time_cpu = time2 - time1  
        print('pt_infer_cpu time cost is %lf seconds' % coast_time_cpu)
        # print(pred_cpu.shape)
        # cv2.imshow('pred_cpu', pred)
        return pred_cpu, coast_time_cpu


    def compile_model(self):
        self.engine = build_engine_from_onnx(args.onnx)
        serialize_engine_to_file(self.engine, args.savepth)
        print('Model compiled and saved.')

    def run_inference(self,frame):
        # ret, frame = self.cap.read()
        # time1 = time.time()
        # print(self.engine_r.get_binding_shape(0)[3], self.engine_r.get_binding_shape(0)[2])
        # img = cv2.resize(frame, (self.engine_r.get_binding_shape(0)[3], self.engine_r.get_binding_shape(0)[2])).astype(np.float32)
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = (img / 255.0 - 0.5) / 0.5
        img = get_image(frame)
        img = np.ascontiguousarray(img)
        # np.copyto(self.h_input, img.ravel())
        time1 = time.time()

        cuda.memcpy_htod_async(self.d_input, img, self.stream)
        self.context.execute_async(bindings=[int(self.d_input)] + [int(d) for d in self.d_outputs], stream_handle=self.stream.handle)
        for h_output, d_output in zip(self.h_outputs, self.d_outputs):
            cuda.memcpy_dtoh_async(h_output, d_output, self.stream)
        self.stream.synchronize()
        pred = self.h_outputs[0].reshape(self.engine_r.get_binding_shape(1))[0, :, :]
        # print(pred.shape)
        time2 = time.time()
        coast_time_trt = time2 - time1  
        print('trt_infer time cost is %lf seconds' % coast_time_trt)
        pred_trt = np.where(pred > 0, 255, 0).astype(np.uint8)
        return pred_trt, coast_time_trt
        # cv2.imshow('trt_pred', pred)
        # row, col = np.nonzero(pred)
        # max_col = max(col[row == 1]) 
        # offset = self.OFFSET / 640 * max_col
        # offset = round(offset, 2) * 100
 

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

    def infer_all(self):
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        # image_paths = [os.path.join(args.impth, filename) for filename in os.listdir(args.impth) if filename.endswith(('.jpg', '.png', '.jpeg'))]

        # for image_path in image_paths:
            
        #     img = get_image(image_path)
        #     # img = np.ascontiguousarray(img)
        #     pred_trt, trt_time = self.run_inference(img)
        #     cv2.imshow('trt_pred', pred_trt)
        #     cv2.waitKey(0)
        while True:
            ret, frame = self.cap.read()
            print('frame shape:', frame.shape)
            pred_trt, trt_time = self.run_inference(frame)
            pred_cuda, cuda_time = self.pt_infer_cuda(frame)
            pred_cpu, cpu_time = self.pt_infer_cpu(frame)
            # print('pred_cpu shape: ', pred_cpu.shape)
            
            # print(np.max(pred_cpu))

            cv2.putText(pred_trt, f"TRT Time: {trt_time:.4f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(pred_cuda, f"CUDA Time: {cuda_time:.4f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(pred_cpu, f"CPU Time: {cpu_time:.4f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            max_height = max(pred_trt.shape[0], pred_cuda.shape[0], pred_cpu.shape[0])
            # print('max_height', max_height)
            pred_trt = cv2.resize(pred_trt, (int(pred_trt.shape[1] * max_height / pred_trt.shape[0]), max_height))
            pred_cuda = cv2.resize(pred_cuda, (int(pred_cuda.shape[1] * max_height / pred_cuda.shape[0]), max_height))
            pred_cpu = cv2.resize(pred_cpu, (int(pred_cpu.shape[1] * max_height / pred_cpu.shape[0]), max_height))
            
            # blank_column = 255 * np.ones((max_height, 10), dtype=np.uint8)  # 使用白色填充
            # print(blank_column.shape)   
        
            # concatenated_image = np.hstack([pred_trt, blank_column, pred_cuda, blank_column, pred_cpu])
            # # concatenated_image = cv2.hconcat([pred_trt, pred_cuda, pred_cpu])
            # h, w = concatenated_image.shape
            # bytesPerLine = 1 * w
            # pred_cpu = pred_cpu[..., None]
            # pred_cuda = pred_cuda[..., None]
            # pred_trt = pred_trt[..., None]
            pred_cpu = cv2.cvtColor(pred_cpu, cv2.COLOR_GRAY2BGR)
            pred_cuda = cv2.cvtColor(pred_cuda, cv2.COLOR_GRAY2BGR)
            pred_trt = cv2.cvtColor(pred_trt, cv2.COLOR_GRAY2BGR)
            # print(pred_cpu.shape)

            cpu_img_add = cv2.addWeighted(frame, 0.8, pred_cpu, 0.5, 0)
            cuda_img_add = cv2.addWeighted(frame, 0.8, pred_cuda, 0.5, 0)
            trt_img_add = cv2.addWeighted(frame, 0.8, pred_trt, 0.5, 0)
            # cv2.imshow('CPU', cpu_img_add)
            # cv2.waitKey(0)
            img_cpu = QImage(cpu_img_add, cpu_img_add.shape[1], cpu_img_add.shape[0], QImage.Format_RGB888)
            q_img = img_cpu.scaled(self.cpu_label.width(), self.cpu_label.height(), Qt.KeepAspectRatio)
            self.cpu_label.setPixmap(QPixmap.fromImage(q_img))

            img_cuda = QImage(cuda_img_add, pred_cpu.shape[1], pred_cpu.shape[0], QImage.Format_RGB888)
            q_img_cuda = img_cuda.scaled(self.cuda_label.width(), self.cuda_label.height(), Qt.KeepAspectRatio)
            self.cuda_label.setPixmap(QPixmap.fromImage(q_img_cuda))

            img_trt = QImage(trt_img_add, pred_cpu.shape[1], pred_cpu.shape[0], QImage.Format_RGB888)
            q_img_trt = img_trt.scaled(self.trt_label.width(), self.trt_label.height(), Qt.KeepAspectRatio)
            self.trt_label.setPixmap(QPixmap.fromImage(q_img_trt))
            # 更新界面
            QApplication.processEvents()


            # # 显示拼接后的图像
            # cv2.imshow('Inference Results', concatenated_image)
            # cv2.waitKey(1)
        # cv2.destroyAllWindows()

        # cv2.imshow('pred_trt', pred_trt)
        # cv2.imshow('pred_cuda', pred_cuda)
        # cv2.imshow('pred_cpu', pred_cpu)
        # cv2.waitKey(0)

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 480)
        # print(self.cap.get(cv2.CAP_PROP_FPS))
        ret, frame = self.cap.read()
        if not ret:
            return
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            scaled_image = image.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(scaled_image))


    def select_save_path(self):
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
        self.cap.release()
        event.accept()
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

