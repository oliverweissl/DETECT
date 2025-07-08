import os

from pygments.styles.dracula import orange

print(os.getcwd())
import torch

import gc

from src_.manipulator_binary import ManipulatorSSpace
from src_.utils import load_generator, load_facial_classifier, load_facial_large_classifier
from configs import (gan_facial_ckpt_path, segmentation_facial_ckpt_path, generate_image_base_dir,
                     sut_facial_path, sut_facial_large_path,
                     preprocess_celeb_classifier, preprocess_celeb_large_classifier,
                     celeba_attributes_dict,
                     )
from local_models.segmentation.segmentation_handler import SegmentationModel



def main(model = 'small'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    # load local_models
    if model == 'small':
        classifier = load_facial_classifier(sut_facial_path, device)
    elif model == 'large':
        classifier = load_facial_large_classifier(sut_facial_large_path, device)
    generator = load_generator(gan_facial_ckpt_path, device)

    """if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        generator = nn.DataParallel(generator)
        classifier = nn.DataParallel(classifier)"""
    segmenter = SegmentationModel(segmentation_facial_ckpt_path)




    target_logit = 15  # glasses
    top_channels = 10 # num of top channels considered in a layer
    extent_factor = 10
    config = "occlusion" # "gradient" or "smoothgrad" or "occlusion"
    oracle = 'confidence_drop'  # or 'misclassification'

    base_dir = os.path.join(generate_image_base_dir, 'runs', f'{model}_{config}_{oracle}')
    os.makedirs(base_dir, exist_ok=True)
    data_path = os.path.join(base_dir, f"{target_logit}_{celeba_attributes_dict[target_logit]}")
    os.makedirs(data_path, exist_ok=True)

    manipulator = ManipulatorSSpace(
        generator=generator,
        classifier=classifier,
        segmenter=segmenter,
        preprocess_fn= preprocess_celeb_classifier if model == 'small' else preprocess_celeb_large_classifier,
        target_logit=target_logit,
        save_dir=data_path,
        device=device
    )
    # generate one random seed from z latent space
    for torch_seed in range(0,21):
        manipulator.handle_one_seed(
            torch_seed=torch_seed,
            top_channels=top_channels,
            default_extent_factor=extent_factor,
            tolerance_of_extent_bisection=1,
            confidence_drop_threshold=0.5,
            oracle=oracle,
            specified_layer=None,
            config= config, # or
        )
        gc.collect()


if __name__ == "__main__":
    main()