import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
import cv2

class SegmentationModel:
    def __init__(self, model_path, device=None):
        """
        Initialize the segmentation model.

        Args:
            model_path (str): Path to the segmentation model checkpoint.
            device (str or torch.device, optional): The device to run the model on. Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = self._load_model(model_path)
        self.preprocess = self._get_preprocessing()
        self.part_labels = self._get_part_labels()
        print(f"Segmentation model loaded on {self.device}.")

    def _load_model(self, model_path):
        """
        Load the segmentation model from the given checkpoint path.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            torch.nn.Module: The loaded segmentation model.
        """
        from local_models.segmentation.bisenet import BiSeNet
        n_classes = 19
        model = BiSeNet(n_classes=n_classes).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _get_part_labels(self):
        """
        Default class labels.

        Returns:
            list: Names of the segmentation parts.
        """
        return [
            "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear",
            "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l",
            "cloth", "hair", "hat"
        ]

    def _get_merged_labels(self):
        """
        Merged class labels.

        Returns:
            dict: Merged class labels.
        """
        return {
            "background": [0],
            "skin": [1, 10, 14, 7, 8, 9],
            "eyebrows": [2, 3],
            "eyes": [4, 5],
            "glasses": [6],
            #"ears": [7, 8, 9], # "earrings": [9],
            "nose": [10],
            "mouth": [11, 12, 13], #"lips": [12, 13],
            # "neck": [14],
            #"necklaces": [15],
            "cloth": [16],
            "hair": [17],
            "hat": [18],
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
    from scipy.stats import wasserstein_distance


    def compute_content_diff(self,img1, img2, mask1, mask2, small_region_threshold=50):

        # 1. Calculate the intersection over union between two binary masks.
        union = np.logical_or(mask1, mask2)
        intersection = np.logical_and(mask1, mask2)
        if union.sum() <= small_region_threshold:
            return -1, -1, -1, -1, "not existing"  # ignore small regions

        iou = intersection.sum() / union.sum()
        iou_diff =1.0 - iou  # 0 = same, 1 = totally different
        if iou<0.1 and mask2.sum() < small_region_threshold:
            change = "removed"
        elif iou<0.1 and mask1.sum() < small_region_threshold:
            change = "added"
        elif iou> 0.8:
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

        # MSE diff
        #mse = ((region1_pixels - region2_pixels) ** 2).mean()

        # 3. Calculate the HSV shift
        hsv1 = cv2.cvtColor((region1 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor((region2 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h1 = hsv1[..., 0]
        h2 = hsv2[..., 0]
        h_diff = np.abs(h1 - h2)
        h_diff = np.minimum(h_diff, 180 - h_diff)
        h_diff = h_diff[union_mask > 0].mean()/90 /2

        # 4. Calculate SSIM (use union region bounding box)
        yx = np.argwhere(union_mask)
        y0, x0 = yx.min(axis=0)
        y1, x1 = yx.max(axis=0) + 1

        crop1 = region1[y0:y1, x0:x1]
        crop2 = region2[y0:y1, x0:x1]

        try:
            ssim_val = ssim(crop1, crop2, channel_axis=-1, data_range=1.0)
        except:
            ssim_val = 1.0  # identical fallback
        ssim_diff = 1.0 - ssim_val

        return iou_diff, l1_diff, h_diff, ssim_diff, change #mse,

    def compute_content_diff_simplified(self,img1, img2, mask1, mask2, small_region_threshold=50):
        # 1. Calculate the intersection over union between two binary masks.
        union_mask = np.logical_or(mask1, mask2).astype(np.uint8)
        area_weight = union_mask.sum() / union_mask.size
        intersection = np.logical_and(mask1, mask2)
        if union_mask.sum() <= small_region_threshold:
            #return -1, [-1,-1,-1 ],-1 , "not existing"  # ignore small regions
            return {
                "iou_diff": -1,
                "hsv_diff": [-1,-1,-1],
                "ssim_diff": -1,
                "change": "not existing",
                "area_weight": area_weight
            }
        iou = intersection.sum() / union_mask.sum()
        iou_diff =1.0 - iou  # 0 = same, 1 = totally different
        if iou<0.1 and mask2.sum() < small_region_threshold:
            change = "removed"
        elif iou<0.1 and mask1.sum() < small_region_threshold:
            change = "added"
        elif iou> 0.7:
            change = "maintained"
        else:
            change = "deformatted"

        if change == "maintained":
            # 2. Calculate SSIM (use union region bounding box)
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
                ssim_val = 1.0  # identical fallback
            ssim_diff = 1.0 - ssim_val

            # 3. Calculate the HSV shift when mask is maintained
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
            if h_diff > 15 or s_diff > 30 or v_diff > 30:
                change = "color changed"
        else:
            h_diff = 0
            s_diff = 0
            v_diff = 0
            ssim_diff = 0
        #return iou_diff, [h_diff/180, s_diff/255, v_diff/255], ssim_diff, change #mse,

        return {
            "iou_diff": iou_diff,
            "hsv_diff": [h_diff/90, s_diff/255, v_diff/255],
            "ssim_diff": ssim_diff,
            "change": change,
            "area_weight": area_weight
        }

    def detect_changes(self,img1, img2, mask1, mask2):
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

            # Composite score (adjustable weight)
            composite_score = (
                0.4 * iou_diff +
                0.1 * ssim_diff +
                0.4 * h_diff +
                0.05 * s_diff +
                0.05 * v_diff
            ) # TO x weight

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


    def _get_preprocessing(self):
        """
        Define the preprocessing pipeline for input images.

        Returns:
            torchvision.transforms.Compose: A transformation pipeline for preprocessing inputs.
        """
        from configs import ToTensorIfNeeded
        return transforms.Compose([
            ToTensorIfNeeded(),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def predict(self, image, resize_shape=(1024,1024)):
        """
        Perform segmentation on the input image.

        Args:
            image (PIL.Image or numpy.ndarray): Input image.
            resize_shape
        Returns:
            torch.Tensor: The segmentation mask.
        """
        # Apply preprocessing

        input_image = self.preprocess(image).unsqueeze(0).to(self.device)

        # Forward pass through the model
        with torch.no_grad():
            output = self.model(input_image)[0]
            # print(output.shape) # 1, 19, 512, 512
            out_max = torch.argmax(output, dim=1)   # Get the highest probability class for each pixel
            out_resize = F.interpolate(out_max.unsqueeze(0).float(),
                                       size=resize_shape,
                                       mode='nearest').squeeze().long()
        return out_resize.cpu().numpy()

    def compute_region_means(self, img, mask):
        """
        Compute the mean values of the given image in each region defined by the segmentation mask.

        Parameters:
            img (numpy.ndarray): The input image with shape (H, W, C), where H = height, W = width,
                                 and C = number of channels.
            mask (numpy.ndarray): The segmentation mask with shape (H, W), where each unique integer
                                  represents a distinct region.

        Returns:
            dict: A dictionary where the keys are region labels (from the mask) and the values are
                  arrays with the mean color for each region (shape = [channels]).
        """
        # Ensure that the mask and image dimensions match
        if img.shape[:2] != mask.shape:
            raise ValueError("The shape of the mask does not match the spatial dimensions of the image.")

        # Dictionary to store the mean value for each region
        region_means = {}

        # Iterate over all unique region labels in the mask
        unique_labels = np.unique(mask)
        for label in unique_labels:
            # Identify pixels corresponding to the current region
            region_pixels = img[mask == label]

            if region_pixels.size == 0:  # Handle edge case
                raise ValueError(f"Region {label} contains no pixels.")

            # Compute the norm for all pixels and all channels in the region
            region_norm = np.linalg.norm(region_pixels)

            # Normalize by the number of pixels to get the average intensity
            average_intensity = region_norm / region_pixels.shape[0]

            region_means[label] = average_intensity  # Store the mean value in the dictionary

        return region_means

    def visualize(self, image, parsing):
        """
        Display the original image, segmentation mask, and overlay.

        Args:
            image (numpy.ndarray): Original image in HWC format with values in [0, 255].
            parsing (numpy.ndarray): Parsed segmentation mask ([H, W]).
        """
        parsing = parsing.squeeze()
        # print(np.unique(parsing))
        # print(len(self.part_labels))
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        color_mask = np.zeros((*parsing.shape, 3), dtype=np.uint8)  # Initialize color mask [H, W, 3]
        for class_idx, color in enumerate(part_colors):
            color_mask[parsing == class_idx] = color

        # Plot results
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(color_mask)
        plt.title("Color Segmentation")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(color_mask, alpha=0.3)
        plt.title("Overlay")
        plt.axis("off")

        # add legends
        legend_patches = []
        for idx, color in enumerate(part_colors):
            if idx in np.unique(parsing):
                patch = mpatches.Patch(color=np.array(color) / 255.0, label=f"{idx}: {self.part_labels[idx]}")
                legend_patches.append(patch)

        plt.legend(
            handles = legend_patches,
            bbox_to_anchor = (1.05, 1),  # Position legend outside the plot
            loc = 'upper left',
            title = "Segmentation Parts")

        plt.tight_layout()
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Initialize the segmentation model
    import os
    print(os.getcwd())
    model_path = "face_segmentation.pth"

    segmenter = SegmentationModel(model_path)

    # Load an example image (replace with the actual image)
    from PIL import Image

    img = Image.open("../../temp_img.png").convert("RGB")

    # Perform segmentation
    mask = segmenter.predict(img)

    # Overlay mask on the original image
    img_np = np.array(img)
    segmenter.visualize(img_np, mask)