import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image


class YOLOSegmentationModel:
    def __init__(self, model_path, device=None, conf_threshold=0.5):
        """
        Initialize the YOLO segmentation model.

        Args:
            model_path (str): Path to the YOLO model checkpoint (.pt file).
            device (str or torch.device, optional): The device to run the model on. Defaults to 'cuda' if available, otherwise 'cpu'.
            conf_threshold (float): Confidence threshold for detections.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.conf_threshold = conf_threshold
        self.model = self._load_model(model_path)
        self.part_labels = self._get_part_labels()
        print(f"YOLO segmentation model loaded on {self.device}.")

    def _load_model(self, model_path):
        """
        Load the YOLO segmentation model from the given checkpoint path.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            YOLO model: The loaded YOLO segmentation model.
        """
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.to(self.device)
            return model
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")

    def _get_part_labels(self):
        """
        Get class labels from the YOLO model.

        Returns:
            list: Names of the segmentation classes.
        """
        if hasattr(self.model, 'names'):
            return list(self.model.names.values())
        else:
            # Fallback to generic labels if model names not available
            return [f"class_{i}" for i in range(80)]  # COCO has 80 classes by default
        # {0: 'back_bumper', 1: 'back_door', 2: 'back_glass', 3: 'back_left_door',
        # 4: 'back_left_light', 5: 'back_light', 6: 'back_right_door',
        # 7: 'back_right_light', 8: 'front_bumper', 9: 'front_door',
        # 10: 'front_glass', 11: 'front_left_door', 12: 'front_left_light',
        # 13: 'front_light', 14: 'front_right_door', 15: 'front_right_light',
        # 16: 'hood', 17: 'left_mirror', 18: 'object', 19: 'right_mirror',
        # 20: 'tailgate', 21: 'trunk', 22: 'wheel'}

    def _get_merged_labels(self):
        """
        Define merged class labels based on your specific use case.
        Modify this according to your fine-tuned model's classes.

        Returns:
            dict: Merged class labels.
        """
        # Example mapping - modify according to your model's classes
        return {
            "background": [18],  # object class as background
            "bumpers": [0, 8],  # back_bumper, front_bumper
            "doors": [1, 3, 6, 9, 11, 14],
            # back_door, back_left_door, back_right_door, front_door, front_left_door, front_right_door
            "glass": [2, 10],  # back_glass, front_glass
            "lights": [4, 5, 7, 12, 13, 15],
            # back_left_light, back_light, back_right_light, front_left_light, front_light, front_right_light
            "body_panels": [16, 20, 21],  # hood, tailgate, trunk
            "mirrors": [17, 19],  # left_mirror, right_mirror
            "wheels": [22],  # wheel
        }

    def get_merged_mask(self, seg_mask, class_ids, use_torch=False):
        """
        Get the merged mask for the given class IDs.

        Args:
            seg_mask (torch.Tensor or numpy.ndarray): The segmentation mask.
            class_ids (list): List of class IDs to merge.
            use_torch (bool, optional): Whether to use PyTorch tensors. Defaults to False.

        Returns:
            torch.Tensor or numpy.ndarray: The merged mask.
        """
        if use_torch:
            return torch.isin(seg_mask, torch.tensor(class_ids)).int()
        else:
            return np.isin(seg_mask, class_ids).astype(np.uint8)

    def compute_content_diff(self, img1, img2, mask1, mask2, small_region_threshold=50):
        """
        Compute content differences between two images based on masks.
        (Same implementation as original)
        """
        # 1. Calculate the intersection over union between two binary masks.
        union = np.logical_or(mask1, mask2)
        intersection = np.logical_and(mask1, mask2)
        if union.sum() <= small_region_threshold:
            return -1, -1, -1, -1, "not existing"

        iou = intersection.sum() / union.sum()
        iou_diff = 1.0 - iou
        if iou < 0.1 and mask2.sum() < small_region_threshold:
            change = "removed"
        elif iou < 0.1 and mask1.sum() < small_region_threshold:
            change = "added"
        elif iou > 0.8:
            change = "maintained"
        else:
            change = "deformatted"

        # 2. Calculate the L1 difference between the two images
        union_mask = np.logical_or(mask1, mask2).astype(np.uint8)
        region1 = img1 * union_mask[..., None]
        region2 = img2 * union_mask[..., None]

        region1_pixels = region1[union_mask > 0]
        region2_pixels = region2[union_mask > 0]

        # L1 diff
        l1_diff = np.abs(region1_pixels - region2_pixels).mean()

        # 3. Calculate the HSV shift
        hsv1 = cv2.cvtColor((region1 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor((region2 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h1 = hsv1[..., 0]
        h2 = hsv2[..., 0]
        h_diff = np.abs(h1 - h2)
        h_diff = np.minimum(h_diff, 180 - h_diff)
        h_diff = h_diff[union_mask > 0].mean() / 90 / 2

        # 4. Calculate SSIM
        yx = np.argwhere(union_mask)
        y0, x0 = yx.min(axis=0)
        y1, x1 = yx.max(axis=0) + 1

        crop1 = region1[y0:y1, x0:x1]
        crop2 = region2[y0:y1, x0:x1]

        try:
            ssim_val = ssim(crop1, crop2, channel_axis=-1, data_range=1.0)
        except:
            ssim_val = 1.0
        ssim_diff = 1.0 - ssim_val

        return iou_diff, l1_diff, h_diff, ssim_diff, change

    def compute_content_diff_simplified(self, img1, img2, mask1, mask2, small_region_threshold=50):
        """
        Simplified content difference computation.
        (Same implementation as original)
        """
        union_mask = np.logical_or(mask1, mask2).astype(np.uint8)
        area_weight = union_mask.sum() / union_mask.size
        intersection = np.logical_and(mask1, mask2)

        if union_mask.sum() <= small_region_threshold:
            return {
                "iou_diff": -1,
                "hsv_diff": [-1, -1, -1],
                "ssim_diff": -1,
                "change": "not existing",
                "area_weight": area_weight
            }

        iou = intersection.sum() / union_mask.sum()
        iou_diff = 1.0 - iou

        if iou < 0.1 and mask2.sum() < small_region_threshold:
            change = "removed"
        elif iou < 0.1 and mask1.sum() < small_region_threshold:
            change = "added"
        elif iou > 0.7:
            change = "maintained"
        else:
            change = "deformatted"

        if change == "maintained":
            region1 = img1 * union_mask[..., None]
            region2 = img2 * union_mask[..., None]

            yx = np.argwhere(union_mask)
            y0, x0 = yx.min(axis=0)
            y1, x1 = yx.max(axis=0) + 1

            crop1 = region1[y0:y1, x0:x1]
            crop2 = region2[y0:y1, x0:x1]

            try:
                ssim_val = ssim(crop1, crop2, channel_axis=-1, data_range=1.0)
            except:
                ssim_val = 1.0
            ssim_diff = 1.0 - ssim_val

            hsv1 = cv2.cvtColor((region1 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor((region2 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

            h1 = hsv1[..., 0][union_mask > 0]
            h2 = hsv2[..., 0][union_mask > 0]
            h_diff = wasserstein_distance(h1, h2)

            s1 = hsv1[..., 1][union_mask > 0]
            s2 = hsv2[..., 1][union_mask > 0]
            s_diff = wasserstein_distance(s1, s2)

            v1 = hsv1[..., 2][union_mask > 0]
            v2 = hsv2[..., 2][union_mask > 0]
            v_diff = wasserstein_distance(v1, v2)
        else:
            h_diff = 0
            s_diff = 0
            v_diff = 0
            ssim_diff = 0

        return {
            "iou_diff": iou_diff,
            "hsv_diff": [h_diff / 90, s_diff / 255, v_diff / 255],
            "ssim_diff": ssim_diff,
            "change": change,
            "area_weight": area_weight
        }

    def detect_changes(self, img1, img2, mask1, mask2):
        """
        Detect changes between two images using their segmentation masks.
        (Same implementation as original)
        """
        result = []
        label_dict = self._get_merged_labels()

        if isinstance(mask1, torch.Tensor):
            mask1 = mask1.cpu().numpy()
        if isinstance(mask2, torch.Tensor):
            mask2 = mask2.cpu().numpy()

        for name, ids in label_dict.items():
            m1 = self.get_merged_mask(mask1, ids)
            m2 = self.get_merged_mask(mask2, ids)

            content_diff = self.compute_content_diff_simplified(img1, img2, m1, m2)
            iou_diff = content_diff["iou_diff"]
            [h_diff, s_diff, v_diff] = content_diff["hsv_diff"]
            ssim_diff = content_diff["ssim_diff"]
            change = content_diff["change"]
            area_weight = content_diff["area_weight"]

            composite_score = (
                    0.4 * iou_diff +
                    0.1 * ssim_diff +
                    0.4 * h_diff +
                    0.05 * s_diff +
                    0.05 * v_diff
            )

            difference_details = {
                "region": name,
                "change": change,
                "iou_diff": round(iou_diff, 2),
                "hsv_shift": f"{round(h_diff, 2)} | {round(s_diff, 2)} | {round(v_diff, 2)}",
                "ssim_diff": round(ssim_diff, 2),
                "composite_score": round(composite_score, 4),
                "weighted_score": round(composite_score * (0.5 + area_weight), 4)
            }
            result.append(difference_details)

        result.sort(key=lambda x: x["weighted_score"], reverse=True)
        return result

    def predict(self, image, resize_shape=None, return_full_results=False):
        """
        Perform segmentation on the input image using YOLO.

        Args:
            image (PIL.Image or numpy.ndarray or str): Input image or path to image.
            resize_shape (tuple): Target size for the output mask.
            return_full_results (bool): Whether to return full YOLO results.

        Returns:
            numpy.ndarray: The segmentation mask or full results if requested.
        """

        from configs import ToTensorIfNeeded
        image = ToTensorIfNeeded()(image)

        # Get original image size
        original_size = image.size  # (width, height)

        # Determine target size
        if resize_shape is None:
            target_size = (original_size[1], original_size[0])  # (height, width)
        else:
            target_size = resize_shape[::-1] if len(resize_shape) == 2 else resize_shape  # Ensure (height, width)

        # Perform prediction
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            save=False,
            verbose=False
        )

        if return_full_results:
            return results

        # Create segmentation mask
        mask = np.zeros(target_size, dtype=np.uint8)  # (height, width)

        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            # Resize masks to target shape
            for i, (seg_mask, class_id) in enumerate(zip(masks, classes)):
                if resize_shape is not None:
                    # Resize mask to target shape
                    resized_mask = cv2.resize(
                        seg_mask.astype(np.uint8),
                        (target_size[1], target_size[0]),  # cv2.resize expects (width, height)
                        interpolation=cv2.INTER_NEAREST
                    )

                # Apply mask with class ID (later masks overwrite earlier ones)
                mask[resized_mask > 0.5] = class_id + 1  # +1 to avoid background class 0

        return mask

    def compute_region_means(self, img, mask):
        """
        Compute the mean values of the given image in each region defined by the segmentation mask.
        (Same implementation as original)
        """
        if img.shape[:2] != mask.shape:
            raise ValueError("The shape of the mask does not match the spatial dimensions of the image.")

        region_means = {}
        unique_labels = np.unique(mask)

        for label in unique_labels:
            region_pixels = img[mask == label]

            if region_pixels.size == 0:
                raise ValueError(f"Region {label} contains no pixels.")

            region_norm = np.linalg.norm(region_pixels)
            average_intensity = region_norm / region_pixels.shape[0]
            region_means[label] = average_intensity

        return region_means

    def visualize(self, image, parsing, show_legend=True):
        """
        Display the original image, segmentation mask, and overlay.

        Args:
            image (numpy.ndarray): Original image in HWC format with values in [0, 255].
            parsing (numpy.ndarray): Parsed segmentation mask ([H, W]).
            show_legend (bool): Whether to show the legend.
        """
        parsing = parsing.squeeze()

        # Generate colors for each class
        unique_classes = np.unique(parsing)
        part_colors = []
        for i in range(len(self.part_labels)):
            # Generate distinct colors
            hue = (i * 137.5) % 360  # Golden angle for better distribution
            saturation = 70 + (i % 3) * 15
            value = 80 + (i % 2) * 20

            # Convert HSV to RGB
            hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
            part_colors.append(rgb_color.tolist())

        color_mask = np.zeros((*parsing.shape, 3), dtype=np.uint8)
        for class_idx in unique_classes:
            if class_idx > 0:  # Skip background
                actual_class = class_idx - 1  # Convert back to 0-based indexing
                if actual_class < len(part_colors):
                    color_mask[parsing == class_idx] = part_colors[actual_class]

        # Plot results
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(color_mask)
        plt.title("YOLO Segmentation")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(color_mask, alpha=0.3)
        plt.title("Overlay")
        plt.axis("off")

        # Add legends
        if show_legend:
            legend_patches = []
            for class_idx in unique_classes:
                if class_idx > 0:  # Skip background
                    actual_class = class_idx - 1
                    if actual_class < len(self.part_labels) and actual_class < len(part_colors):
                        patch = mpatches.Patch(
                            color=np.array(part_colors[actual_class]) / 255.0,
                            label=f"{class_idx}: {self.part_labels[actual_class]}"
                        )
                        legend_patches.append(patch)

            if legend_patches:
                plt.legend(
                    handles=legend_patches,
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    title="Detected Classes"
                )

        plt.tight_layout()
        plt.show()


# Example Usage
if __name__ == "__main__":
    import os
    print(os.getcwd())

    os.chdir("../..")
    # Initialize the YOLO segmentation model
    model_path = "local_models/segmentation/carparts_segment/carparts_seg_4gpu/weights/best.pt"  # Replace with your model path

    segmenter = YOLOSegmentationModel(model_path, conf_threshold=0.5)

    # Load an example image
    img_path = "stylegan_car.png"  # Replace with your image path
    img = Image.open(img_path).convert("RGB")

    # Perform segmentation
    mask = segmenter.predict(img)

    # Visualize results
    img_np = np.array(img)
    segmenter.visualize(img_np, mask)

    # Print detected classes
    unique_classes = np.unique(mask)
    print("Detected classes:", unique_classes)
    for class_id in unique_classes:
        if class_id > 0:  # Skip background
            actual_class = class_id - 1
            if actual_class < len(segmenter.part_labels):
                print(f"Class {class_id}: {segmenter.part_labels[actual_class]}")