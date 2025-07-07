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
    extent_factor = 80
    truncation_psi = 0.6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    # load local_models
    generator = load_generator(gan_car_ckpt_path, device)
    yolo = YOLO("yolov8n.pt")
    yolo_model = yolo.model.to(device)

    """if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        generator = nn.DataParallel(generator)
        classifier = nn.DataParallel(classifier)"""
    segmenter = None

    base_dir = os.path.join(generate_image_base_dir, f'generated_car_images_yolo_{extent_factor}_{truncation_psi}')
    os.makedirs(base_dir, exist_ok=True)

    top_channels = 2

    manipulator = ManipulatorSSpace(
        generator=generator,
        classifier=yolo_model,
        segmenter=segmenter,
        #target_class=2,
        # preprocess_fn=preprocess_celeb_classifier,
        save_dir="",
        device=device
    )
    # generate one random seed from z latent space
    for torch_seed in range(0,100):

        data_path = os.path.join(base_dir, f"{torch_seed}")
        manipulator.save_dir = data_path
        # os.makedirs(data_path, exist_ok=True)

        manipulator.handle_one_seed(
            torch_seed=torch_seed,
            class_dict=yolo.names,
            top_channels=top_channels,
            default_extent_factor=extent_factor,
            tolerance_of_extent_bisection=1,
            confidence_drop_threshold=0.3,
            specified_layer=None,
            truncation_psi= truncation_psi
        )
        gc.collect()


if __name__ == "__main__":
    main()