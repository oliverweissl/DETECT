import os
import torch
import gc
import argparse
from ultralytics import YOLO

# Facial recognition imports
from src.s_manipulator_binary import BinaryManipulatorSSpace as FacialManipulatorSSpace
from src.utils import load_generator, load_facial_classifier, load_facial_large_classifier

# Dog classification imports
from src.s_manipulator_multi import MulticlassManipulatorSSpace as DogManipulatorSSpace
from src.utils import load_rexnet_dog_classifier

# YOLO object detection imports
from src.s_manipulator_yolo import ObjectDetectionManipulatorSSpace as YoloManipulatorSSpace



from configs import (
    gan_facial_ckpt_path, gan_lsun_dog_ckpt_path, gan_car_ckpt_path,
    generate_image_base_dir,
    sut_facial_path, sut_facial_large_path, sut_dog_path,
    preprocess_celeb_classifier, preprocess_celeb_large_classifier,
    celeba_attributes_dict, dog_rexnet_dict
)


def run_facial_detection(args):
    """Run facial attribute detection"""
    print("Running facial attribute detection...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    # Load models
    if args.model == 'small':
        classifier = load_facial_classifier(sut_facial_path, device)
        preprocess_fn = preprocess_celeb_classifier
    elif args.model == 'large':
        classifier = load_facial_large_classifier(sut_facial_large_path, device)
        preprocess_fn = preprocess_celeb_large_classifier
    else:
        raise ValueError("Model must be 'small' or 'large' for facial detection")

    generator = load_generator(gan_facial_ckpt_path, device)
    segmenter = None

    base_dir = os.path.join(generate_image_base_dir, 'runs_', f'{args.model}_{args.config}_{args.oracle}')
    os.makedirs(base_dir, exist_ok=True)
    data_path = os.path.join(base_dir, f"{args.target_logit}_{celeba_attributes_dict[args.target_logit]}")
    os.makedirs(data_path, exist_ok=True)

    manipulator = FacialManipulatorSSpace(
        generator=generator,
        classifier=classifier,
        segmenter=segmenter,
        preprocess_fn=preprocess_fn,
        target_logit=args.target_logit,
        save_dir=data_path,
        confidence_drop_threshold=args.confidence_threshold,
        device=device
    )

    # Generate seeds
    for torch_seed in range(args.start_seed, args.end_seed):
        print(f"torch_seed {torch_seed}")
        manipulator.handle_one_seed(
            torch_seed=torch_seed,
            default_extent_factor=args.extent_factor,
            oracle=args.oracle,
            specified_layer=None,
            truncation_psi=args.truncation_psi,
            config=args.config,
        )
        gc.collect()


def run_dog_classification(args):
    """Run dog breed classification"""
    print("Running dog breed classification...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    # Load models
    generator = load_generator(gan_lsun_dog_ckpt_path, device)
    classifier = load_rexnet_dog_classifier(sut_dog_path, device)
    segmenter = None

    base_dir = os.path.join(generate_image_base_dir, 'runs', f'abc_dogs_{args.config}_{args.oracle}')
    os.makedirs(base_dir, exist_ok=True)

    manipulator = DogManipulatorSSpace(
        generator=generator,
        classifier=classifier,
        segmenter=segmenter,
        preprocess_fn=preprocess_celeb_classifier,
        save_dir="",
        class_dict=dog_rexnet_dict,
        confidence_drop_threshold=args.confidence_threshold,
        device=device
    )

    # Generate seeds
    for torch_seed in range(args.start_seed, args.end_seed):
        data_path = os.path.join(base_dir, f"{torch_seed}")
        manipulator.save_dir = data_path

        manipulator.handle_one_seed(
            torch_seed=torch_seed,
            default_extent_factor=args.extent_factor,
            oracle=args.oracle,
            specified_layer=None,
            skip_rgb_layer=True,
            truncation_psi=args.truncation_psi,
            config=args.config
        )
        gc.collect()


def run_yolo_detection(args):
    """Run YOLO object detection"""
    print("Running YOLO object detection...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    # Load models
    generator = load_generator(gan_car_ckpt_path, device)
    yolo = YOLO("yolov8n.pt")
    yolo_model = yolo.model.to(device)
    segmenter = None

    base_dir = os.path.join(generate_image_base_dir, 'runs', f'abc_yolocar_{args.config}_{args.oracle}')
    os.makedirs(base_dir, exist_ok=True)

    manipulator = YoloManipulatorSSpace(
        generator=generator,
        classifier=yolo_model,
        segmenter=segmenter,
        save_dir=base_dir,
        class_dict=yolo.names,
        confidence_drop_threshold=args.confidence_threshold,
        device=device
    )

    # Generate seeds
    for torch_seed in range(args.start_seed, args.end_seed):
        manipulator.handle_one_seed(
            torch_seed=torch_seed,
            default_extent_factor=args.extent_factor,
            oracle=args.oracle,
            specified_layer=None,
            truncation_psi=args.truncation_psi,
            config=args.config
        )
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Unified main script for facial, dog, and YOLO detection')

    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=['facial', 'dog', 'yolo'],
                        help='Task to run: facial, dog, or yolo')

    # Model configuration
    parser.add_argument('--model', type=str, default='large',
                        choices=['small', 'large'],
                        help='Model size for facial detection (default: large)')

    # Processing configuration
    parser.add_argument('--config', type=str, default='smoothgrad',
                        choices=['gradient', 'smoothgrad', 'occlusion'],
                        help='Attribution method (default: smoothgrad)')

    parser.add_argument('--oracle', type=str, default='confidence_drop',
                        choices=['confidence_drop', 'misclassification'],
                        help='Oracle type (default: confidence_drop)')

    # Parameters
    parser.add_argument('--extent_factor', type=int, default=10,
                        help='Extent factor (recommended: 10 - 20)')

    parser.add_argument('--truncation_psi', type=float, default=0.7,
                        help='Truncation psi value (default: 0.7)')

    parser.add_argument('--confidence_threshold', type=float, default=0.4,
                        help='Confidence drop threshold (default: 0.4)')

    parser.add_argument('--target_logit', type=int, default=15,
                        help='Target logit for facial detection (default: 15)')

    # Seed range
    parser.add_argument('--start_seed', type=int, default=0,
                        help='Starting seed number (default: 0)')

    parser.add_argument('--end_seed', type=int, default=100,
                        help='Ending seed number (default: 100)')

    args = parser.parse_args()

    # Auto-adjust extent_factor based on oracle if not explicitly set
    if args.extent_factor == 10 and args.oracle == 'misclassification':
        args.extent_factor = 20
        print(f"Auto-adjusted extent_factor to {args.extent_factor} for misclassification oracle")

    # Auto-adjust truncation_psi for YOLO if not explicitly set
    if args.task == 'yolo' and args.truncation_psi == 0.7:
        args.truncation_psi = 0.5
        print(f"Auto-adjusted truncation_psi to {args.truncation_psi} for YOLO task")

    # Auto-adjust seed ranges based on original scripts
    if args.start_seed == 0 and args.end_seed == 100:
        if args.task == 'facial':
            args.end_seed = 78
        elif args.task == 'dog':
            args.start_seed = 10
            args.end_seed = 50
        print(f"Auto-adjusted seed range to [{args.start_seed}, {args.end_seed}) for {args.task} task")

    print(f"Running {args.task} task with configuration:")
    print(f"  Model: {args.model if args.task == 'facial' else 'N/A'}")
    print(f"  Config: {args.config}")
    print(f"  Oracle: {args.oracle}")
    print(f"  Extent factor: {args.extent_factor}")
    print(f"  Truncation psi: {args.truncation_psi}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  Target logit: {args.target_logit if args.task == 'facial' else 'N/A'}")
    print(f"  Seed range: [{args.start_seed}, {args.end_seed})")

    # Run the appropriate task
    if args.task == 'facial':
        run_facial_detection(args)
    elif args.task == 'dog':
        run_dog_classification(args)
    elif args.task == 'yolo':
        run_yolo_detection(args)


if __name__ == "__main__":
    main()
