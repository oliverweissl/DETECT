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

    truncation_psi = 0.7
    segmenter = None
    extent_factor = 20  # 10 for confidence_drop and 20 for misclassification
    top_channels = 10
    config = "smoothgrad" # "gradient" or "smoothgrad"
    oracle = 'misclassification'  # 'confidence_drop'  or 'misclassification'

    base_dir = os.path.join(generate_image_base_dir, 'runs', f'dogs_{config}_{oracle}')
    os.makedirs(base_dir, exist_ok=True)
    manipulator = ManipulatorSSpace(
        generator=generator,
        classifier=classifier,
        segmenter=segmenter,
        preprocess_fn=preprocess_celeb_classifier,
        save_dir="",
        device=device
    )
    # generate one random seed from z latent space
    for torch_seed in range(10,50):

        data_path = os.path.join(base_dir, f"{torch_seed}")
        manipulator.save_dir = data_path
        # os.makedirs(data_path, exist_ok=True)

        manipulator.handle_one_seed(
            torch_seed=torch_seed,
            class_dict=dog_rexnet_dict,
            top_channels=top_channels,
            default_extent_factor=extent_factor,
            confidence_drop_threshold=0.3,
            oracle= oracle, # "misclassification"
            specified_layer=None,
            skip_rgb_layer=True,
            truncation_psi = truncation_psi,
            config = config
        )
        gc.collect()


if __name__ == "__main__":
    main()