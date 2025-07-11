import os
print(os.getcwd())
import torch

import gc
from ultralytics import YOLO
from src_.manipulator_yolo import ManipulatorSSpace
from src_.utils import load_generator
from configs import (gan_car_ckpt_path, generate_image_base_dir)
#from local_models.segmentation.segmentation_handler import SegmentationModel



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    # load local_models
    generator = load_generator(gan_car_ckpt_path, device)
    yolo = YOLO("yolov8n.pt")
    yolo_model = yolo.model.to(device)

    truncation_psi = 0.5
    segmenter = None
    extent_factor = 20  # 10 for confidence_drop and 20 for misclassification
    top_channels = 10
    config = "smoothgrad" # "gradient" or "smoothgrad"
    oracle = 'misclassification'  # 'confidence_drop'  or 'misclassification'

    base_dir = os.path.join(generate_image_base_dir, 'runs', f'yolocar_{config}_{oracle}')
    os.makedirs(base_dir, exist_ok=True)

    manipulator = ManipulatorSSpace(
        generator=generator,
        classifier=yolo_model,
        segmenter=segmenter,
        save_dir=base_dir,
        device=device
    )
    # generate one random seed from z latent space
    for torch_seed in range(0,100):
        # os.makedirs(data_path, exist_ok=True)

        manipulator.handle_one_seed(
            torch_seed=torch_seed,
            class_dict=yolo.names,
            top_channels=top_channels,
            default_extent_factor=extent_factor,
            confidence_drop_threshold=0.4,
            oracle=oracle,
            specified_layer=None,
            truncation_psi= truncation_psi,
            config = config

        )
        gc.collect()


if __name__ == "__main__":
    main()