from ultralytics import YOLO
from tqdm.notebook import tqdm
import os
import cv2
import torch
from torchvision import transforms
import numpy as np
from dataclasses import dataclass, field
import json
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time


@dataclass
class SoccerNetJerseyPreprocessor:
    root: str
    output: str
    model_seg = YOLO("yolo11n-seg.pt")  # pretrained YOLO11n model
    model_pose = YOLO("yolo11n-pose.pt")  # pretrained YOLO11n model

    dbscan_config = {"eps": 0.2, "min_samples": 4}
    imgsz_seg = (512, 256)
    imgsz_pose = (512, 256)
    yolo_batch = 32
    max_det = 4
    expand_scale = (1.65, 1.20)
    ratio_hw_thres = 0.4
    ratio_tot_thres = 0.2
    face_conf_thres = 0.78

    idxs: list = field(default_factory=list, init=False)
    yolo_results: list = field(default_factory=list, init=False)
    file_paths: list = field(default_factory=list, init=False)

    def __len__(self):
        return len(os.listdir(self.root))

    def read_sorted_files(self, idx):
        path = os.path.join(self.root, str(idx))
        files = os.listdir(path)
        files.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
        full_paths = [os.path.join(path, f) for f in files]
        return full_paths

    def apply_filterer(self, filterer):
        assert len(self.idxs) == len(filterer)
        self.idxs = [i for i, ok in zip(self.idxs, filterer) if ok]
        self.yolo_results = [r for r, ok in zip(self.yolo_results, filterer) if ok]
        self.file_paths = [f for f, ok in zip(self.file_paths, filterer) if ok]

    def cluster_images(self, features):
        dbscan = DBSCAN(**self.dbscan_config)
        labels = dbscan.fit_predict(features)
        unique_labels, counts = np.unique(labels, return_counts=True)
        positive_label = unique_labels[np.argmax(counts)]  # the major cluster
        return [label == positive_label for label in labels]

    def get_img_mask(self, seg_result):
        image = torch.from_numpy(seg_result.orig_img)
        mask_full = seg_result.masks[0][0].data.clone().bool().cpu()
        mask = transforms.Resize(image.shape[:2])(mask_full).squeeze(0)
        return image, mask

    def __call__(self, data_no, show_final_result=0, output=True):
        self.file_paths = self.read_sorted_files(data_no)
        self.idxs = list(range(len(self.file_paths)))
        # segment the images
        self.yolo_results = self.model_seg(
            self.file_paths,
            imgsz=self.imgsz_seg,
            batch=self.yolo_batch,
            classes=[0],
            half=True,
            max_det=self.max_det,
            save=False,
            stream=False,
        )
        # print(len(self.file_paths))
        # print(len(self.idxs))
        # print(len(self.yolo_results))
        self.apply_filterer([len(r.boxes) == 1 for r in self.yolo_results])
        features = []
        for result in self.yolo_results:
            if len(result.boxes) == 1:
                image, mask = self.get_img_mask(result)
                msk_uint8 = mask.byte().numpy() * 255
                hist_img = cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist(
                    [hist_img], [0], msk_uint8, [16], [0, 180]
                ).T.flatten()
                hist = hist / np.sum(hist)  # L1 normalization
                features.append(hist)
        if len(features) == 0:
            return
        is_major_cluster = self.cluster_images(features)
        self.apply_filterer(is_major_cluster)

        self.yolo_results = self.model_pose(
            self.file_paths,
            imgsz=self.imgsz_pose,
            batch=self.yolo_batch,
            half=True,
            save=False,
            stream=False,
        )

        # 0. Nose           # 1. Left Eye       # 2. Right Eye      # 3. Left Ear
        # 4. Right Ear      # 5. Left Shoulder  # 6. Right Shoulder # 7. Left Elbow
        # 8. Right Elbow    # 9. Left Wrist     # 10. Right Wrist   # 11. Left Hip
        # 12. Right Hip     # 13. Left Knee     # 14. Right Knee    # 15. Left Ankle
        # # 16. Right Ankle

        four_points_expd = []
        is_cropped_number = []
        for result in self.yolo_results:
            keypoints = result.keypoints
            is_cropped_number.append(False)
            if keypoints.has_visible and len(keypoints.xy) == 1:
                xy = keypoints.xy.squeeze(0).cpu()
                points4 = xy[[5, 6, 11, 12]]
                # if any points is (0,0), skip it
                if any(torch.all(p == 0) for p in points4):
                    continue
                ls, rs, lh, rh = points4
                width = torch.dist(ls, rs)
                height = (torch.dist(ls, lh) + torch.dist(rs, rh)) / 2
                ratio = width / height
                ratio_tot = height / result.orig_img.shape[0]
                face_conf = keypoints.conf[:3].mean()
                if (
                    ratio > self.ratio_hw_thres
                    and ratio_tot > self.ratio_tot_thres
                    and face_conf < self.face_conf_thres
                ):
                    print(face_conf)
                    center = points4.mean(0)
                    scale = torch.Tensor(self.expand_scale)  # x sacle, y scale
                    points4_expand = (points4 - center) * scale + center
                    for d in 0, 1:
                        points4_expand[d, :] = torch.clamp(
                            points4_expand[d, :], 0, result.orig_img.shape[d] - 1
                        )
                    four_points_expd.append(points4_expand)
                    is_cropped_number[-1] = True  # cropped
        self.apply_filterer(is_cropped_number)

        crop_imgs = []
        for result, points4 in zip(self.yolo_results, four_points_expd):
            img = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
            points = np.array([points4[[0, 1, 3, 2]]], dtype=np.int32)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, points, 255)
            mean_color = cv2.mean(img, mask=mask)
            img[mask == 0, :] = mean_color[:3]
            cv2.fillPoly(mask, points, (255, 255, 255))

            minx = points4[:, 0].min().int()
            maxx = points4[:, 0].max().int()
            miny = points4[:, 1].min().int()
            maxy = points4[:, 1].max().int()

            crop_img = img[miny : maxy + 1, minx : maxx + 1, :]
            crop_imgs.append(crop_img)

            if show_final_result > 0:
                show_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                plt.imshow(show_img)
                plt.show()
        
        if not output:
            return crop_imgs
        
        output_path = os.path.join(self.output, str(data_no))
        os.makedirs(output_path, exist_ok=True)
        for i, crop_img in enumerate(crop_imgs):
            output_file = os.path.join(output_path, f"{i}.bmp")
            print(output_file)
            cv2.imwrite(output_file, crop_img)