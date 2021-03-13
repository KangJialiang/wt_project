import glob
import json
import os
import queue
import random
import threading
import time
from ctypes import *

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

import camera_parameters as params
import darknet.darknet as darknet
import monodepth2.networks as networks
from camera_parameters import *
from deep_sort import build_tracker
from deep_sort.parser import get_config
from kalman_filter import KalmanFilter
from monodepth2.layers import disp_to_depth

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CUDA NOT AVALIABLE")


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=10,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class CameraCapture(cv2.VideoCapture):
    """Bufferless & Distorted VideoCapture"""

    def __init__(self, original_options: tuple, intrinsic_matrix, dist_coeffs):
        super().__init__(*original_options)
        # self._queue = queue.SimpleQueue()
        self._queue = queue.Queue()
        read_camera_thread = threading.Thread(target=self._reader)
        read_camera_thread.daemon = True
        read_camera_thread.start()

        self._intrinsic_matrix = intrinsic_matrix.cpu().numpy()
        self._dist_coeffs = np.array(dist_coeffs)
        frame = self._queue.get()
        self._new_intrinsic_matrix, self._new_xywh = cv2.getOptimalNewCameraMatrix(
            self._intrinsic_matrix, self._dist_coeffs, frame.shape[:2], 0)

        self.intrinsic_matrix = torch.tensor(
            self._new_intrinsic_matrix).to(device)

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            self._success, frame = super().read()
            if not self._success:
                break
            while True:
                try:
                    self._queue.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    break
            self._queue.put(frame)

    def _distort_img(self, img):
        distorted_img = cv2.undistort(img, self._intrinsic_matrix, self._dist_coeffs,
                                      None, self._new_intrinsic_matrix)
        x, y, w, h = self._new_xywh
        distorted_img = distorted_img[x:x+w, y:y+h]
        return distorted_img

    def read(self):
        return self._success, self._distort_img(self._queue.get())


class FileCapture():
    def __init__(self, file_path: str, ext="jpg") -> None:
        images_list = glob.glob(f'{file_path}/*.{ext}')
        images_list.sort(key=lambda x: int(x[len(file_path)+1:-len(ext)-1]))
        self.images = iter(images_list)

        self.intrinsic_matrix = torch.FloatTensor(
            [[785.26446533, 0., 627.50964355],
             [0., 785.27935791, 340.54248047],
             [0.,    0.,  1.]]).to(device)

    def release(self):
        pass

    def read(self):
        success_flag = False
        try:
            fname = next(self.images)
            frame = cv2.imread(fname)
            success_flag = True
        except:
            frame = None
            pass

        return success_flag, frame


class Trajectory():
    def __init__(self, max_age=50, max_error=0.1):
        self.max_age = max_age
        self.max_error = max_error

        self.objects = dict()
        self.index = 0

    def __delete_out_dated(self):
        to_be_deleted = []

        for obj_id, coords_dict in self.objects.items():
            last_index = max([key for key in coords_dict.keys()])
            if self.index-last_index > self.max_age:
                to_be_deleted.append(obj_id)

        for index in to_be_deleted:
            self.objects.pop(index)

    def update(self, coords, ids):
        self.__delete_out_dated()

        for i, id in enumerate(ids):
            if id not in self.objects.keys():
                self.objects[id] = {self.index: coords[i]}
            else:
                if self.index-1 in self.objects[id]:
                    self.objects[id][self.index] = coords[i]
                else:
                    last_index = max([key for key in self.objects[id].keys()])
                    for index in range(last_index+1, self.index+1):
                        last_coord = self.objects[id][last_index]
                        current_coord = coords[i]
                        self.objects[id][index] = [last_coord[coord]+(current_coord[coord]-last_coord[coord])*(
                            index-last_index)/(self.index-last_index) for coord in range(len(coords[i]))]

        self.index += 1

    def get_nearest(self, distance) -> dict:
        distance_dict = dict()
        min_delta = float("inf")
        for obj_id, coords_dict in self.objects.items():
            last_index = max([key for key in coords_dict.keys()])
            current_coord = coords_dict[last_index]
            current_distance = (sum(x**2 for x in current_coord))**.5
            distance_dict[obj_id] = current_distance
        for obj_id, obj_distance in distance_dict.items():
            if abs(obj_distance-distance) < min_delta:
                min_delta = abs(obj_distance-distance)
                best_match = obj_id
        if min_delta < self.max_error:
            return self.objects[best_match]
        return None


def init_monodepth_model(model_name):
    """Function to predict for a single image or folder of images
    """

    model_path = os.path.join("./monodepth2/models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    return feed_height, feed_width, encoder, depth_decoder


def get_relative_depth(frame, feed_height, feed_width, encoder, depth_decoder):
    input_image = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        # Load image and preprocess
        original_width, original_height = input_image.size
        input_image = input_image.resize(
            (feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)  # 插值成源图像大小

        _, depth = disp_to_depth(disp_resized, 0.1, 100)

        return depth


def pixelcoord_to_worldcoord(depth_matrix, intrinsic_matrix, inv_extrinsics_matrix, pixel_indexs):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    v = pixel_indexs[0, :]
    u = pixel_indexs[1, :]

    depth_vector = depth_matrix.view(-1)

    v = (v-cy)*depth_vector/fy
    u = (u-cx)*depth_vector/fx

    ones = torch.ones(depth_vector.size()).to(device)

    P_cam = torch.stack((u, v, depth_vector, ones), dim=0)
    # [x: crosswise ,y: -lengthwise, z: vertical, 1]
    P_w = torch.mm(inv_extrinsics_matrix, P_cam)

    # np.savetxt('P_cam.txt', P_cam.cpu().numpy()[:, :10])
    # np.savetxt('P_w.txt', P_w.cpu().numpy()[:, :10])

    return P_w


def get_mask(x_left, y_top, x_right, y_bottom, to_size, portion=1):
    """Edges all included"""

    if portion > 0 and portion < 1:
        mask = torch.bernoulli(
            torch.ones(y_bottom-y_top+1, x_right-x_left+1)*portion)
    else:
        mask = torch.ones(y_bottom-y_top+1, x_right-x_left+1)
    padding = (
        x_left,  # padding in left
        to_size[1]-x_right-1,  # padding in right
        y_top,  # padding in top
        to_size[0]-y_bottom-1  # padding in bottom
    )
    mask = torch.nn.functional.pad(
        mask, padding,  mode="constant", value=0).type(torch.bool).to(device)
    return mask


def find_diff(last_frame, current_frame, threshold=10):
    diff = cv2.absdiff(last_frame, current_frame)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    static_points = torch.from_numpy((diff < threshold)).to(device)

    return static_points


def get_scale(relative_disp: torch.tensor, intrinsic_matrix: torch.tensor, inv_extrinsics_matrix: torch.tensor,
              camera_height: float, pixel_indexs, current_frame, last_frame, last_true_disp, portion=1):
    mask = get_mask(relative_disp.size()[1]*3//8,
                    relative_disp.size()[0]*27//40,
                    relative_disp.size()[1]*5//8,
                    relative_disp.size()[0]*37//40,
                    relative_disp.size(), portion=portion)
    road_points = relative_disp*mask

    P_w = pixelcoord_to_worldcoord(road_points, intrinsic_matrix,
                                   inv_extrinsics_matrix, pixel_indexs)
    rel_heights = torch.masked_select(P_w[2, :], mask.view(-1))  # 选取z坐标

    std = torch.std(rel_heights)
    mean = torch.mean(rel_heights)
    threshold_mask = torch.lt(torch.abs(rel_heights-mean), std)
    rel_heights = torch.masked_select(rel_heights, threshold_mask.view(-1))
    scale_camera_height_based = camera_height / \
        (rel_heights.sum()/rel_heights.shape[0])

    if last_frame is not None and last_true_disp is not None:
        static_points = find_diff(last_frame, current_frame)
        scale_static_points_based = torch.sum(last_true_disp*static_points.reshape_as(last_true_disp)) /\
            torch.sum(relative_disp*static_points.reshape_as(relative_disp))
        scale = .1*scale_camera_height_based+0.9*scale_static_points_based
        # print(torch.sum(static_points) / last_true_disp.numel())
    else:
        scale = scale_camera_height_based

    return scale


def init_darknet_network(config_file: str, data_file: str, weights_file: str,):
    network, class_names, class_colors = darknet.load_network(
        config_file, data_file, weights_file, batch_size=1)
    return network, class_names, class_colors


def detection(darknet_network, class_names, class_colors, frame, confidence_thresh=0.25):
    original_height = frame.shape[0]
    original_width = frame.shape[1]
    network_width = darknet.network_width(darknet_network)
    network_height = darknet.network_height(darknet_network)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (network_width, network_height),
                               interpolation=cv2.INTER_LINEAR)

    img_for_detect = darknet.make_image(network_width, network_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

    detections = darknet.detect_image(
        darknet_network, class_names, img_for_detect, thresh=confidence_thresh)
    darknet.free_image(img_for_detect)
    detections_resized = []
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        bbox = (x*original_width/network_width,
                y*original_height/network_height,
                w*original_width/network_width,
                h*original_height/network_height,)
        detections_resized.append((label, confidence, bbox))

    return detections_resized


def get_coordinates(P_w, outputs, frame):
    measurement_list = []
    id_list = []
    for output in outputs:
        x1, y1, x2, y2, id = output
        # mask = get_mask(x1+(x2-x1)//10, y1+(y2-y1)//10,
        #                 x2-(x2-x1)//10, y2-(y2-y1)//10, frame.shape)
        mask = get_mask(x1, y1, x2, y2, frame.shape)
        x_coords = torch.masked_select(P_w[0, :], mask.view(-1))
        y_coords = torch.masked_select(P_w[1, :], mask.view(-1))
        z_coords = torch.masked_select(P_w[2, :], mask.view(-1))
        coords = torch.stack((x_coords, y_coords, z_coords), dim=0)
        distance_w = torch.norm(coords, p=2, dim=0)
        min_distance = torch.min(distance_w)
        index = (distance_w == min_distance).nonzero().flatten()[0]
        x_distance = float(coords[0, index])
        y_distance = float(coords[1, index])
        z_distance = float(coords[2, index])
        measurement_list.append([x_distance, y_distance, z_distance])
        id_list.append(id)
    return measurement_list, id_list


# @profile
def main():
    # # read form camera
    # intrinsic_matrix = params.intrinsic_matrix
    # camera = CameraCapture((0,), intrinsic_matrix, dist_coeffs)

    # camera = CameraCapture(
    #     (gstreamer_pipeline(), cv2.CAP_GSTREAMER), intrinsic_matrix, dist_coeffs)

    # read form file
    camera = FileCapture("./img")
    intrinsic_matrix = camera.intrinsic_matrix

    # cv2.namedWindow("Test camera")
    # cv2.namedWindow("Result")
    # cv2.namedWindow("MultiTracker")

    # choices = ["mono_640x192",
    #            "stereo_640x192",
    #            "mono+stereo_640x192",
    #            "mono_no_pt_640x192",
    #            "stereo_no_pt_640x192",
    #            "mono+stereo_no_pt_640x192",
    #            "mono_1024x320",
    #            "stereo_1024x320",
    #            "mono+stereo_1024x320"]

    # initiate monodepth
    feed_height, feed_width, encoder, depth_decoder = init_monodepth_model(
        "mono_640x192")

    # initiate yolo
    darknet_network, class_names, class_colors = init_darknet_network(config_file="./darknet/yolo-obj.cfg",
                                                                      data_file="./darknet/data/obj.data",
                                                                      weights_file="./darknet/yolo-obj.weights")

    # initiate deep track
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort = build_tracker(cfg, use_cuda=torch.cuda.is_available())

    # initiate Kalman filter
    measurement_matrix = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32)
    transition_matrix = np.array(
        [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1],
         [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)
    process_noise_cov = np.eye(6, dtype=np.float32) * 1e-3
    measurement_noise_cov = np.eye(3, dtype=np.float32) * 1e-1
    kalman_filter = KalmanFilter(6, 3, measurement_matrix,
                                 transition_matrix, process_noise_cov, measurement_noise_cov)

    # initiate trajectory recorder
    trajectory_recorder = Trajectory()

    success, frame = camera.read()

    pixel_indexs = torch.tensor([[v, u]
                                 for v in range(frame.shape[0])
                                 for u in range(frame.shape[1])]).t().to(device)

    last_frame = None
    last_true_disp = None

    last_y = dict()
    time_serial_index = 0

    while success:
        last_run_time = time.time()

        # key = cv2.waitKey(1)
        # # if key == 27 or not success or\
        # #         cv2.getWindowProperty("Test camera", cv2.WND_PROP_AUTOSIZE) < 1 or\
        # #         cv2.getWindowProperty("Result", cv2.WND_PROP_AUTOSIZE) < 1 or\
        # #         cv2.getWindowProperty("MultiTracker", cv2.WND_PROP_AUTOSIZE) < 1:
        # if key == 27 or not success:
        #     break
        # if key == ord(']'):
        #     continue

        # do depth estimation
        rel_disp = get_relative_depth(
            frame,  feed_height, feed_width, encoder, depth_decoder)
        scale = get_scale(rel_disp, intrinsic_matrix,
                          inv_extrinsics_matrix, camera_height, pixel_indexs,
                          frame, last_frame, last_true_disp)
        true_disp = rel_disp*scale
        P_w = pixelcoord_to_worldcoord(true_disp, intrinsic_matrix,
                                       inv_extrinsics_matrix, pixel_indexs)
        last_frame = frame
        last_true_disp = true_disp

        # do detection
        detections = detection(
            darknet_network, class_names, class_colors, frame)
        detections = np.array(detections, dtype=object)
        if detections.size > 0:
            bbox_xywh = np.array([np.array(xywh) for xywh in detections[:, 2]])
            cls_conf = detections[:, 1].astype(np.float)
        else:
            bbox_xywh = np.array([[], [], [], []]).T
            cls_conf = np.array([[], [], [], []]).T

        # do tracking
        outputs = deepsort.update(bbox_xywh, cls_conf, frame)

        # get coordinates
        coords, ids = get_coordinates(P_w, outputs, frame)

        # do kalman filting
        filtered = kalman_filter.update(coords, ids)

        # record trajectory
        trajectory_recorder.update([filtered[index] for index in ids], ids)

        # # get depth image
        # disp_resized_np = -P_w[1, :].cpu().numpy().reshape(
        #     frame.shape[0], frame.shape[1])
        # # Saving colormapped depth image
        # vmax = np.percentile(disp_resized_np, 95)
        # # normalizer = mpl.colors.Normalize(
        # #     vmin=disp_resized_np.min(), vmax=vmax)
        # normalizer = mpl.colors.Normalize(vmin=0, vmax=20)
        # # print(f"min: {disp_resized_np.min()}\tmax: {vmax}")
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        # colormapped_im = (mapper.to_rgba(disp_resized_np)[
        #     :, :, :3] * 255).astype(np.uint8)
        # im = pil.fromarray(colormapped_im)
        # # im.save(f'./temp/{temp_index}_disp.jpg')

        # plot result
        font = cv2.FONT_HERSHEY_DUPLEX
        font_thickness = 1
        for output in outputs:
            x1, y1, x2, y2, id = output
            random.seed(id)
            color = (random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          color, 2)

            x_distance, y_distance, z_distance = filtered[id]

            if id in last_y:
                speed = (last_y[id]+y_distance)/0.1244
            else:
                speed = 0
            last_y[id] = -y_distance
            text_line1 = f"y:{-y_distance:0.2f}m"
            text_line2 = f"speed:{speed:0.2f}m/s"
            font_scale = 0.8
            cv2.putText(frame, text_line1, (x2, y1), font,
                        font_scale, color, font_thickness, cv2.LINE_AA)
            size_line1 = cv2.getTextSize(
                text_line1, font, font_scale, font_thickness)[0]
            cv2.putText(frame, text_line2, (x2, y1+size_line1[1]), font,
                        font_scale, color, font_thickness, cv2.LINE_AA)

        font_scale = 2
        fps_text = f"FPS:{1/(time.time()-last_run_time):0.1f}"
        size_fps_text = cv2.getTextSize(
            fps_text, font, font_scale, font_thickness)[0]
        cv2.putText(frame, fps_text, (frame.shape[0]-size_fps_text[0], size_fps_text[1]), font,
                    font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        # cv2.imshow("MultiTracker", frame)
        # cv2.imwrite(f'./temp/{temp_index}.jpg', frame)

        print(f"index: {time_serial_index}")

        # if time_serial_index == 50:
        #     break

        success, frame = camera.read()
        time_serial_index += 1

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
