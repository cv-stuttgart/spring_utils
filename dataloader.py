import torch.utils.data as data
import numpy as np
from glob import glob
import os
from PIL import Image
import flow_IO


class SpringFlowDataset(data.Dataset):
    """
    Dataset class for Spring optical flow dataset.
    For train, this dataset returns image1, image2, flow and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    For test, this dataset returns image1, image2 and a data tuple (framenum, scene name, left/right cam, FW/BW direction).

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """
    def __init__(self, root, split='train', subsample_groundtruth=True):
        super(SpringFlowDataset, self).__init__()

        assert split in ["train", "test"]
        seq_root = os.path.join(root, split)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []

        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                # forward
                for frame in range(1, len(images)):
                    self.data_list.append((frame, scene, cam, "FW"))
                # backward
                for frame in reversed(range(2, len(images)+1)):
                    self.data_list.append((frame, scene, cam, "BW"))

    def __getitem__(self, index):
        frame_data = self.data_list[index]
        frame, scene, cam, direction = frame_data

        img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")

        if direction == "FW":
            img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame+1:04d}.png")
        else:
            img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame-1:04d}.png")

        img1 = np.asarray(Image.open(img1_path)) / 255.0
        img2 = np.asarray(Image.open(img2_path)) / 255.0

        if self.split == "test":
            return img1, img2, frame_data

        flow_path = os.path.join(self.seq_root, scene, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{frame:04d}.flo5")
        flow = flow_IO.readFlowFile(flow_path)
        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> flow will have same dimensions as images
            flow = flow[::2,::2]

        return img1, img2, flow, frame_data

    def __len__(self):
        return len(self.data_list)


class SpringStereoDataset(data.Dataset):
    """
    Dataset class for Spring stereo dataset.
    For train, this dataset returns image1, image2, disparity and a data tuple (framenum, scene name, left/right cam).
    image1 is the reference frame; image2 is the image from the other camera.
    For test, this dataset returns image1, image2 and a data tuple (framenum, scene name, left/right cam).

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """
    def __init__(self, root, split='train', subsample_groundtruth=True):
        super(SpringStereoDataset, self).__init__()

        assert split in ["train", "test"]
        seq_root = os.path.join(root, split)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []

        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                for frame in range(1, len(images)+1):
                    self.data_list.append((frame, scene, cam))

    def __getitem__(self, index):
        frame_data = self.data_list[index]
        frame, scene, cam = frame_data

        img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")
        if cam == "left":
            othercam = "right"
        else:
            othercam = "left"
        img2_path = os.path.join(self.seq_root, scene, f"frame_{othercam}", f"frame_{othercam}_{frame:04d}.png")

        img1 = np.asarray(Image.open(img1_path)) / 255.0
        img2 = np.asarray(Image.open(img2_path)) / 255.0

        if self.split == "test":
            return img1, img2, frame_data

        disp_path = os.path.join(self.seq_root, scene, f"disp1_{cam}", f"disp1_{cam}_{frame:04d}.dsp5")
        disp = flow_IO.readDispFile(disp_path)
        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> disp will have same dimensions as images
            disp = disp[::2,::2]

        if cam == "left":
            # left-to-right disparity should be negative, which is not encoded in the data
            disp *= -1

        return img1, img2, disp, frame_data

    def __len__(self):
        return len(self.data_list)


class SpringSceneFlowDataset(data.Dataset):
    """
    Dataset class for Spring scene flow dataset.
    For train, this dataset returns image1, image2, image3, image4, disp1, disp2, flow and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    For test, this dataset returns image1, image2, image3, image4 and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    The images are:
    image1: reference frame
    image2: same time step as reference frame, but other camera
    image3: next/previous time step compared to reference frame, same camera as reference frame
    image4: other time step and other camera than reference frame

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """
    def __init__(self, root, split='train', subsample_groundtruth=True):
        super(SpringSceneFlowDataset, self).__init__()

        assert split in ["train", "test"]
        seq_root = os.path.join(root, split)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []

        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                # forward
                for frame in range(1, len(images)):
                    self.data_list.append((frame, scene, cam, "FW"))
                # backward
                for frame in reversed(range(2, len(images)+1)):
                    self.data_list.append((frame, scene, cam, "BW"))

    def __getitem__(self, index):
        frame_data = self.data_list[index]
        frame, scene, cam, direction = frame_data

        # reference frame
        img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")
        if cam == "left":
            othercam = "right"
        else:
            othercam = "left"

        if direction == "FW":
            othertimestep = frame+1
        else:
            othertimestep = frame-1

        # same time step, other cam
        img2_path = os.path.join(self.seq_root, scene, f"frame_{othercam}", f"frame_{othercam}_{frame:04d}.png")
        # other time step, same cam
        img3_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{othertimestep:04d}.png")
        # other time step, other cam
        img4_path = os.path.join(self.seq_root, scene, f"frame_{othercam}", f"frame_{othercam}_{othertimestep:04d}.png")

        img1 = np.asarray(Image.open(img1_path)) / 255.0
        img2 = np.asarray(Image.open(img2_path)) / 255.0
        img3 = np.asarray(Image.open(img3_path)) / 255.0
        img4 = np.asarray(Image.open(img4_path)) / 255.0

        if self.split == "test":
            return img1, img2, img3, img4, frame_data

        disp1_path = os.path.join(self.seq_root, scene, f"disp1_{cam}", f"disp1_{cam}_{frame:04d}.dsp5")
        disp1 = flow_IO.readDispFile(disp1_path)
        disp2_path = os.path.join(self.seq_root, scene, f"disp2_{direction}_{cam}", f"disp2_{direction}_{cam}_{frame:04d}.dsp5")
        disp2 = flow_IO.readDispFile(disp2_path)
        flow_path = os.path.join(self.seq_root, scene, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{frame:04d}.flo5")
        flow = flow_IO.readFlowFile(flow_path)
        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> ground truth will have same dimensions as images
            disp1 = disp1[::2,::2]
            disp2 = disp2[::2,::2]
            flow = flow[::2,::2]

        return img1, img2, img3, img4, disp1, disp2, flow, frame_data

    def __len__(self):
        return len(self.data_list)


def main():
    """
    Test all data loaders
    """
    spring_root = os.getenv("SPRING_DIR", "/data/spring")
    print("looking for Spring dataset in", spring_root)

    spring_flow_test = SpringFlowDataset(root=spring_root, split="test")
    assert len(spring_flow_test) == 3960
    for i in range(3960):
        print(spring_flow_test[i][-1])

    spring_flow_train = SpringFlowDataset(root=spring_root, split="train")
    assert len(spring_flow_train) == 19852
    for i in range(19852):
        print(spring_flow_train[i][-1])

    spring_sceneflow_test = SpringSceneFlowDataset(root=spring_root, split="test")
    assert len(spring_sceneflow_test) == 3960
    for i in range(3960):
        print(spring_sceneflow_test[i][-1])

    spring_sceneflow_train = SpringSceneFlowDataset(root=spring_root, split="train")
    assert len(spring_sceneflow_train) == 19852
    for i in range(19852):
        print(spring_sceneflow_train[i][-1])

    spring_stereo_test = SpringStereoDataset(root=spring_root, split="test")
    assert len(spring_stereo_test) == 2000
    for i in range(2000):
        print(spring_stereo_test[i][-1])

    spring_stereo_train = SpringStereoDataset(root=spring_root, split="train")
    assert len(spring_stereo_train) == 10000
    for i in range(10000):
        print(spring_stereo_train[i][-1])


if __name__ == "__main__":
    main()