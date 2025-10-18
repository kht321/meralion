"""
Compare ASR Multimodal LLM Toxicity Evaluation Results (Visualization Only)
"""

import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")


def plot_visualizations(df_summary, output_dir):
    """Create plots\."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_context("talk")

    datasets = df_summary["Dataset"].unique()

    # --- Plot 1: WER and CER boxplots (excluding Text input) ---
    df_asr = df_summary[df_summary["Input Type"].str.lower() != "text"]
    for metric, label in [("Average WER", "WER"), ("Average CER", "CER")]:
        fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 8), sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, datasets):
            subset = df_asr[df_asr["Dataset"] == dataset]
            sns.boxplot(data=subset, x="Model", y=metric, hue="Input Type", ax=ax)
            ax.set_title(f"{dataset}")
            ax.set_xlabel(" ")
            ax.set_ylabel(metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", 
                   ncol=len(labels), bbox_to_anchor=(0.5, 0.01))
        for ax in axes:
            ax.get_legend().remove()
        plt.suptitle(f"Transcription {label}")
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(os.path.join(output_dir, f"transcription_{label.lower()}.png"))
        plt.close(fig)

    # --- Plot 2: Toxicity Classification Accuracy ---
    df_acc = df_summary[~df_summary["Model"].str.lower().str.contains("whisper")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        subset = df_acc[df_acc["Dataset"] == dataset]
        sns.barplot(data=subset, x="Model", y="Classification Accuracy", hue="Input Type", ax=ax)
        ax.set_title(f"{dataset}")
        ax.set_xlabel(" ")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.01))
    for ax in axes:
        ax.get_legend().remove()
    plt.suptitle("Toxicity Classification Accuracy")
    plt.tight_layout(rect=[0, 0.08, 1, 1]) 
    plt.savefig(os.path.join(output_dir, "classification_accuracy.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Toxicity Classification F1 ---
    df_f1 = df_summary[~df_summary["Model"].str.lower().str.contains("whisper")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=True)
    axes = axes.flatten()
    f1_types = ["F1 Toxic", "F1 Non-Toxic"]
    datasets_order = ["Test", "Trigger Test"]
    for i, (f1_col, dataset) in enumerate(
        [(f, d) for d in datasets_order for f in f1_types]
    ):
        ax = axes[i]
        subset = df_f1[df_f1["Dataset"] == dataset]
        sns.barplot(data=subset, x="Model", y=f1_col, hue="Input Type", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(f"{dataset} – {f1_col[3:]}")
        ax.set_xlabel(" ")
        ax.set_ylabel("F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(labels), bbox_to_anchor=(0.5, 0.05))
    for ax in axes:
        ax.get_legend().remove()
    plt.suptitle("Toxicity Classification F1 Score")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(output_dir, "classification_f1.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Plot 4: Total Blocks by Input Type ---
    df_blocks = df_summary[
        (~df_summary["Model"].str.lower().str.contains("whisper")) &
        (~df_summary["Input Type"].str.lower().isin(["text"]))
    ]

    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        subset = df_blocks[df_blocks["Dataset"] == dataset]
        sns.barplot(data=subset, x="Model", y="Total Blocks", hue="Input Type", ax=ax)
        ax.set_title(f"{dataset}")
        ax.set_xlabel(" ")
        ax.set_ylabel("Block Count")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", 
               ncol=len(labels), bbox_to_anchor=(0.5, 0.01))
    for ax in axes:
        ax.get_legend().remove()
    plt.suptitle("Transcription Block Count")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(output_dir, "transcription_blocks.png"))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for ASR multimodal LLM toxicity evaluation.")
    parser.add_argument("--csv_path", type=str,
                        default="results/toxicity/summary.csv",
                        help="Path to precompiled summary CSV.")
    parser.add_argument("--output_dir", type=str,
                        default="results/toxicity/figures",
                        help="Directory to save visualizations.")
    args = parser.parse_args()

    print(f"Loading summary CSV from: {args.csv_path}")
    df_summary = pd.read_csv(
        args.csv_path, 
        dtype={"Dataset": str, "Model": str, "Input Type": str, "Prompt": str,
               "Average WER": float, "Average CER": float,
               "Classification Accuracy": float, "F1 Toxic": float},
        na_values=["-"]
        )

    print("Generating visualizations")
    plot_visualizations(df_summary, args.output_dir)

    print("Completed!")
    print(f"Plots saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
