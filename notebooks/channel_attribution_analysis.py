import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

# 设置matplotlib样式
plt.style.use('dark_background')


def load_json_data(base_dir):
    """
    Load JSON data from the specified base directory.

    Args:
        base_dir (str): Path to the directory containing seed directories and JSON files.

    Returns:
        pd.DataFrame: A DataFrame containing aggregate data from all JSON files.
    """
    records = []

    # Process all subdirectories
    for seed_dir in os.listdir(base_dir):
        seed_path = os.path.join(base_dir, seed_dir)
        if not os.path.isdir(seed_path):
            print(f"Skipping non-directory: {seed_path}")
            continue

        # Process all JSON files in the subdirectory
        for json_file in os.listdir(seed_path):
            if not json_file.endswith(".json"):
                continue
            json_path = os.path.join(seed_path, json_file)

            try:
                # Load JSON data
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Extract relevant fields
                row = {}
                layer_name = data.get("layer_name")
                channel_id = data.get("channel_id")
                significant_changes = data.get("significant_changes", [])
                row.update({
                    "layer_name": layer_name,
                    "channel_id": channel_id,
                    "img_path": data.get("img_path")
                })

                # Extract up to 2 significant changes
                for i, change in enumerate(significant_changes[:2]):
                    row.update({
                        f"region_{i}": change.get("region"),
                        f"change_{i}": change.get("change"),
                        f"iou_diff_{i}": change.get("iou_diff"),
                        f"hsv_shift_{i}": change.get("hsv_shift"),
                        f"ssim_diff_{i}": change.get("ssim_diff"),
                        f"composite_score_{i}": change.get("composite_score"),
                        f"weighted_score_{i}": change.get("weighted_score")
                    })
                records.append(row)
            except Exception as e:
                print(f"Error reading file {json_path}: {e}")

    return pd.DataFrame(records)


def get_channel_attribution(df, layer_name, channel_id, threshold=0.6):
    """
    Get the attribution for a specific channel based on region+change frequency.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        layer_name (str): The name of the layer.
        channel_id (int): The ID of the channel.
        threshold (float): Threshold for consensus (default: 0.6 for 60%)

    Returns:
        dict: Dictionary containing attribution information
    """
    # Filter for the specified channel
    channel_df = df[(df["layer_name"] == layer_name) & (df["channel_id"] == channel_id)]

    if channel_df.empty:
        return {
            "layer_name": layer_name,
            "channel_id": channel_id,
            "attribution": "NO_DATA",
            "confidence": 0.0,
            "total_samples": 0,
            "top_attribution": None,
            "top_frequency": 0,
            "needs_human_annotation": True
        }

    # Create combined attribution strings for both region_0+change_0 and region_1+change_1
    attributions = []

    # Process region_0 + change_0 and region_1 + change_1
    for _, row in channel_df.iterrows():
        if pd.notna(row.get("region_0")) and pd.notna(row.get("change_0")):
            attribution_0 = f"{row['region_0']}_{row['change_0']}"
            attributions.append(attribution_0)

        if pd.notna(row.get("region_1")) and pd.notna(row.get("change_1")):
            attribution_1 = f"{row['region_1']}_{row['change_1']}"
            attributions.append(attribution_1)

    if not attributions:
        return {
            "layer_name": layer_name,
            "channel_id": channel_id,
            "attribution": "NO_VALID_DATA",
            "confidence": 0.0,
            "total_samples": len(channel_df),
            "top_attribution": None,
            "top_frequency": 0,
            "needs_human_annotation": True
        }

    # Count attribution frequencies
    attribution_counts = Counter(attributions)
    total_attributions = len(attributions)

    # Get top attribution and its frequency
    top_attribution, top_count = attribution_counts.most_common(1)[0]
    top_frequency = top_count / total_attributions

    # Determine attribution based on threshold
    if top_frequency >= threshold:
        final_attribution = top_attribution
        needs_annotation = False
    else:
        final_attribution = "NEEDS_HUMAN_ANNOTATION"
        needs_annotation = True

    return {
        "layer_name": layer_name,
        "channel_id": channel_id,
        "attribution": final_attribution,
        "confidence": top_frequency,
        "total_samples": len(channel_df),
        "total_attributions": total_attributions,
        "top_attribution": top_attribution,
        "top_frequency": top_count,
        "needs_human_annotation": needs_annotation,
        "distribution": dict(attribution_counts)
    }


def analyze_all_channels_attribution(df, threshold=0.6):
    """
    Analyze attribution for all channels in the dataset.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        threshold (float): Threshold for consensus (default: 0.6 for 60%)

    Returns:
        pd.DataFrame: DataFrame with attribution results for all channels
    """
    # Get unique layer-channel combinations
    unique_channels = df[["layer_name", "channel_id"]].drop_duplicates()

    attributions = []

    for _, row in unique_channels.iterrows():
        layer_name = row["layer_name"]
        channel_id = row["channel_id"]

        attribution_info = get_channel_attribution(df, layer_name, channel_id, threshold)
        attributions.append(attribution_info)

    # Convert to DataFrame
    attribution_df = pd.DataFrame(attributions)

    # Sort by confidence (descending) and then by total_samples (descending)
    attribution_df = attribution_df.sort_values(
        by=["confidence", "total_samples"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return attribution_df


def print_attribution_summary(attribution_df):
    """
    Print a summary of the attribution analysis.

    Args:
        attribution_df (pd.DataFrame): DataFrame with attribution results
    """
    total_channels = len(attribution_df)
    auto_attributed = len(attribution_df[attribution_df["needs_human_annotation"] == False])
    needs_annotation = len(attribution_df[attribution_df["needs_human_annotation"] == True])

    print(f"=== Channel Attribution Summary ===")
    print(f"Total channels analyzed: {total_channels}")
    print(f"Automatically attributed: {auto_attributed} ({auto_attributed / total_channels * 100:.1f}%)")
    print(f"Needs human annotation: {needs_annotation} ({needs_annotation / total_channels * 100:.1f}%)")
    print()

    # Show top attributed channels
    print("=== Top Automatically Attributed Channels ===")
    auto_channels = attribution_df[attribution_df["needs_human_annotation"] == False]
    if not auto_channels.empty:
        display_cols = ["layer_name", "channel_id", "attribution", "confidence", "total_samples"]
        print(auto_channels[display_cols].head(10).to_string(index=False))
    else:
        print("No channels meet the threshold for automatic attribution.")

    print()

    # Show channels needing annotation
    print("=== Channels Needing Human Annotation ===")
    need_annotation = attribution_df[attribution_df["needs_human_annotation"] == True]
    if not need_annotation.empty:
        display_cols = ["layer_name", "channel_id", "top_attribution", "confidence", "total_samples"]
        print(need_annotation[display_cols].head(10).to_string(index=False))
    else:
        print("All channels meet the threshold for automatic attribution.")


def plot_channel_samples(df, layer_name, channel_id, sample_count=5, save_path=None):
    """
    Plot sample images for a specific channel for annotation purposes.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        layer_name (str): The name of the layer.
        channel_id (int): The ID of the channel.
        sample_count (int): Number of samples to display.
        save_path (str): Path to save the plot (optional).
    """
    # Filter DataFrame for the specified layer and channel
    filtered_df = df[(df["layer_name"] == layer_name) & (df["channel_id"] == channel_id)]

    if filtered_df.empty:
        print(f"No data found for layer: {layer_name}, channel: {channel_id}")
        return

    # Randomly sample rows
    sampled_rows = filtered_df.sample(n=min(sample_count, len(filtered_df)))

    # Create subplot
    fig, axes = plt.subplots(1, len(sampled_rows), figsize=(4 * len(sampled_rows), 4))
    if len(sampled_rows) == 1:
        axes = [axes]

    fig.suptitle(f"Samples for {layer_name}_channel_{channel_id}", fontsize=16)

    for i, (idx, row) in enumerate(sampled_rows.iterrows()):
        img_path = row["img_path"]
        comparison_path = img_path.replace(".png", "_comparison.png")

        if os.path.exists(comparison_path):
            # Load and crop image
            img = plt.imread(comparison_path)[140:330, 60:600]
            axes[i].imshow(img)

            # Create title with region and change info
            title_parts = []
            if pd.notna(row.get('region_0')) and pd.notna(row.get('change_0')):
                title_parts.append(f"{row['region_0']}_{row['change_0']}")
            if pd.notna(row.get('region_1')) and pd.notna(row.get('change_1')):
                title_parts.append(f"{row['region_1']}_{row['change_1']}")

            axes[i].set_title(' | '.join(title_parts), fontsize=10)
            axes[i].axis("off")
        else:
            axes[i].text(0.5, 0.5, f"Image not found:\n{comparison_path}",
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

    plt.show()


def save_attribution_results(attribution_df, output_path):
    """
    Save attribution results to CSV file.

    Args:
        attribution_df (pd.DataFrame): DataFrame with attribution results
        output_path (str): Path to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    attribution_df.to_csv(output_path, index=False)
    print(f"Attribution results saved to: {output_path}")


def analyze_channels_for_annotation(df, attribution_df, output_dir="./annotation_results"):
    """
    Generate sample images for all channels that need annotation.

    Args:
        df (pd.DataFrame): The original DataFrame containing data.
        attribution_df (pd.DataFrame): DataFrame with attribution results.
        output_dir (str): Directory to save annotation images.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get channels that need annotation
    need_annotation = attribution_df[attribution_df["needs_human_annotation"] == True]

    print(f"Generating sample images for {len(need_annotation)} channels needing annotation...")

    for i, row in need_annotation.iterrows():
        layer_name = row["layer_name"]
        channel_id = row["channel_id"]

        # Generate filename
        filename = f"{layer_name}_channel_{channel_id}_samples.png"
        save_path = os.path.join(output_dir, filename)

        # Plot samples
        print(f"Processing {layer_name}_channel_{channel_id}...")
        plot_channel_samples(df, layer_name, channel_id, sample_count=5, save_path=save_path)

        # Also print distribution info
        print(f"  Distribution: {row['distribution']}")
        print(f"  Top attribution: {row['top_attribution']} (confidence: {row['confidence']:.3f})")
        print()


def main(base_dir, threshold=0.6, output_dir="./analysis_results"):
    """
    Main function to orchestrate the entire analysis process.

    Args:
        base_dir (str): Path to the directory containing JSON files.
        threshold (float): Threshold for automatic attribution (default: 0.6).
        output_dir (str): Directory to save results.
    """
    print("=== Channel Attribution Analysis ===")
    print(f"Base directory: {base_dir}")
    print(f"Threshold: {threshold}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    print("Loading JSON data...")
    df = load_json_data(base_dir)
    print(f"Loaded {len(df)} records")
    print()

    # Analyze attributions
    print("Analyzing channel attributions...")
    attribution_df = analyze_all_channels_attribution(df, threshold=threshold)
    print()

    # Print summary
    print_attribution_summary(attribution_df)
    print()

    # Save results
    results_path = os.path.join(output_dir, "channel_attributions.csv")
    save_attribution_results(attribution_df, results_path)
    print()

    # Generate annotation images
    annotation_dir = os.path.join(output_dir, "annotation_samples")
    analyze_channels_for_annotation(df, attribution_df, annotation_dir)

    print("Analysis complete!")
    return df, attribution_df


if __name__ == "__main__":
    from configs import generate_image_base_dir, celeba_attributes_dict

    target_logit = 15
    base_dir = os.path.join(generate_image_base_dir, 'runs/small_smoothgrad_confidence_drop',
                            f"{target_logit}_{celeba_attributes_dict[target_logit]}")


    df, attribution_results = main(base_dir, threshold=0.6, output_dir="./analysis_results")