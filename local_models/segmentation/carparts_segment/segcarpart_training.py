from ultralytics import YOLO
import torch
import os


# Disable wandb logging to avoid project name issues
os.environ['WANDB_DISABLED'] = 'true'

# Check if CUDA is available and how many GPUs
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Load a pretrained segmentation model like YOLO11n-seg
model = YOLO("carparts_seg_4gpu/weights/best.pt")  # load a pretrained model (recommended for training)


#%%
# Train the model on the Carparts Segmentation dataset with 4 GPUs
results = model.train(
    data="data_file/carparts-seg/data.yaml",
    epochs=100,
    imgsz=640,
    device=[0, 1, 2, 3],  # Use 4 GPUs (GPU 0, 1, 2, 3)
    batch=128,  # Increase batch size for multi-GPU training
    save=True,  # Save training checkpoints
    save_period=10,  # Save checkpoint every 10 epochs
    plots=True,  # Generate training plots
    name="carparts_seg_4gpu",  # Custom experiment name
    project="carparts_segment",  # Project directory (no slashes)
    exist_ok=True,  # Allow overwriting existing experiments
    workers=8,  # Number of data loading workers
    patience=50,  # Early stopping patience
    cache=False,  # Don't cache images for multi-GPU training
    amp=True,  # Automatic Mixed Precision
    cos_lr=False,  # Don't use cosine learning rate scheduler
    resume=False,  # Don't resume from checkpoint
    verbose=True  # Verbose output

)

# After training, you can validate the model's performance on the validation set
print("Starting validation...")

results = model.val(
    data="data_file/carparts-seg/data.yaml",  # Make sure this points to the correct data file
    plots=True,  # Generate validation plots
    save_json=True,  # Save validation results as JSON
    device=0,  # Use only one GPU for validation to avoid memory overflow
    batch=128,  #
    imgsz=640,  # Image size
    workers=4,  # Reduce number of workers
    verbose=True,  # Verbose output
    half=False,  # Don't use half precision to avoid potential issues
    project="carparts_validation",  # Separate project for validation
    name="val_results",  # Validation results folder name
    exist_ok=True,  # Allow overwriting existing results
    split='val',  # Specify validation split
    save_txt=True,  # Save results in txt format
    save_conf=True  # Save confidence scores

)


# Print validation results
print("Validation completed!")
print(f"Validation results: {results}")
if hasattr(results, 'names'):
    print(f"Classes found in validation: {results.names}")

print(f"Validation results saved to: carparts_validation/val_results/")

# Check the data.yaml file to verify class names
try:
    import yaml
    with open("../../../data_file/carparts-seg/data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)
        print(f"Classes in data.yaml: {data_config.get('names', 'Not found')}")
except Exception as e:
    print(f"Could not read data.yaml: {e}")
