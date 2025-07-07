import torch
from torchvision.utils import save_image
import torchvision
import os
from src_.backpropagation import backpropagation_gradients_s_space
import json
from src.manipulator._style_gan_manipulator.legacy import load_network_pkl

def generate_and_record_images(target_class,
                               base_path,
                               num_samples=10,
                               generator_ckpt_path = "local_models/generators/imagenet128.pkl",
                               output_dir="data",
                               device=None):
    """
    Generate images for some classes, predict results, and record gradients.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load networks
    with open(generator_ckpt_path, 'rb') as f:
        generator = load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    mapping_net = generator.mapping
    synthesis_net = generator.synthesis
    classifier = torchvision.models.resnet18(pretrained=True).eval()
    classifier = classifier.to(device)

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # generate class label
    label = torch.zeros([1, generator.c_dim], device=device)
    label[:, target_class] = 1

    # Switch local_models to evaluation mode
    mapping_net.eval()
    synthesis_net.eval()
    classifier.eval()

    # Ensure output directories
    output_dir = os.path.join(base_path, output_dir, str(target_class))
    print(f"absolute path: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)


    for i in range(num_samples):
        print(f"Generating image {i}")
        # Step 1: Generate random latent vector
        z = torch.randn([1, generator.z_dim], device=device)

        # Step 2: Generate image using the synthesis network
        w = mapping_net(z, label)

        w_grad, prediction, img = backpropagation_gradients_s_space(
            synthesis_net=synthesis_net,
            classifier=classifier,
            preprocess=preprocess,
            w_latents=w,
            target_class=target_class,
            device=device)

        # Save the generated image
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
        img_path = os.path.join(output_dir, str(i),f"image_{i}.png")
        save_image(img.squeeze(), img_path)
        print(f"Generated and saved image: {img_path}")

        # Get the top 2 predictions
        top2_probs, top2_classes = torch.topk(prediction, 2)
        print(f"Top 2 predictions: {top2_probs}, {top2_classes}")
        predicted_class, second_predicted_class = top2_classes[0][0], top2_classes[0][1]
        print(f"Predicted class: {predicted_class}")
        # Save prediction data in JSON format
        metadata = {
            "image_id": i,
            "image_path": img_path,
            "predicted_class": int(top2_classes[0][0]),
            "predicted_probability": float(top2_probs[0][0]),
            "second_predicted_class": int(top2_classes[0][1]),
            "second_predicted_probability": float(top2_probs[0][1]),
            "gradient_path": f"gradients/gradient_{i}.npy"
        }
        metadata_path = os.path.join(output_dir, str(i), f"metadata_{i}.json")
        with open(metadata_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)

        # Save gradients
        grad_path = os.path.join(output_dir, str(i), f"gradient_{i}.pt")
        # w_grad_list = w_grad.tolist()
        #with open(grad_path, "w") as json_file:
        #    json.dump(w_grad_list, json_file, indent=4)
        torch.save(w_grad, grad_path)
        del z, w, w_grad, prediction, img
        #torch.cuda.empty_cache()

# Example usage
if __name__ == "__main__":
    # Ensure device compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate and record images, predictions, and gradients
    generate_and_record_images(target_class=207, num_samples=5, device=device)