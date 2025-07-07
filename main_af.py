import os
print(os.getcwd())
import torch

import gc

from src_.manipulator_multiclass import ManipulatorSSpace
from src_.utils import load_generator, load_rexnet_dog_classifier
from configs import (gan_lsun_dog_ckpt_path,
                     generate_image_base_dir,
                     preprocess_celeb_classifier, sut_dog_path,
                     dog_rexnet_dict)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    # load local_models
    generator = load_generator(gan_lsun_dog_ckpt_path, device)
    classifier = load_rexnet_dog_classifier(sut_dog_path, device)
    """if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        generator = nn.DataParallel(generator)
        classifier = nn.DataParallel(classifier)"""
    segmenter = None

    base_dir = os.path.join(generate_image_base_dir, 'generated_dog_images_rexnet')
    os.makedirs(base_dir, exist_ok=True)

    top_channels = 2

    manipulator = ManipulatorSSpace(
        generator=generator,
        classifier=classifier,
        segmenter=segmenter,
        preprocess_fn=preprocess_celeb_classifier,
        save_dir="",
        device=device
    )
    # generate one random seed from z latent space
    for torch_seed in range(50):

        data_path = os.path.join(base_dir, f"{torch_seed}")
        manipulator.save_dir = data_path
        # os.makedirs(data_path, exist_ok=True)

        manipulator.handle_one_seed(
            torch_seed=torch_seed,
            class_dict=dog_rexnet_dict,
            top_channels=top_channels,
            default_extent_factor=0.05,
            tolerance_of_extent_bisection=1,
            confidence_drop_threshold=0.3,
            specified_layer=None,
        )
        gc.collect()


if __name__ == "__main__":
    main()