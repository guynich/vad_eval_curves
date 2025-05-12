"""Test script for voice activity detection (VAD) using the Hugging Face dataset."""

import pprint

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from datasets import load_dataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from speech_detector import SpeechDetector  # Silero VAD model

# SAMPLE_RATE = 16000
SPLITS = ["test.clean", "test.other"]
# LATENCY = 0.5

VAD_THRESHOLD = 0.5


# def calculate_nested_mean(nested_list):
#     """Calculate the mean of all items in a nested list structure."""
#     return np.mean([item for sublist in nested_list for item in sublist])


# def compute_auc(y_true, y_scores):
#     # y_scores: model output probabilities or scores
#     roc_auc = roc_auc_score(y_true, y_scores)
#     pr_auc = average_precision_score(y_true, y_scores)
#     return {"ROC AUC": roc_auc, "PR AUC": pr_auc}


def compute_overall_auc(list_of_y_true, list_of_y_scores):
    # Flatten all true labels and scores
    all_y_true = [label for sublist in list_of_y_true for label in sublist]
    all_y_scores = [score for sublist in list_of_y_scores for score in sublist]

    roc_auc = roc_auc_score(all_y_true, all_y_scores)
    pr_auc = average_precision_score(all_y_true, all_y_scores)

    return {"Overall ROC AUC": roc_auc, "Overall PR AUC": pr_auc}


def evaluate_binary_classification(y_true, y_pred):
    # Ensure inputs are lists of 0s and 1s

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Precision: TP / (TP + FP)
    precision = precision_score(y_true, y_pred, zero_division=0)

    # Recall: TP / (TP + FN)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
    }


# def plot_overall_curves(list_of_y_true, list_of_y_scores):
#     # Flatten all true labels and scores
#     all_y_true = [label for sublist in list_of_y_true for label in sublist]
#     all_y_scores = [score for sublist in list_of_y_scores for score in sublist]

#     # Compute ROC and PR curves
#     fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
#     precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)

#     # Compute AUCs
#     roc_auc = roc_auc_score(all_y_true, all_y_scores)
#     pr_auc = average_precision_score(all_y_true, all_y_scores)

#     # Plot
#     plt.figure(figsize=(12, 5))

#     # ROC
#     plt.subplot(1, 2, 1)
#     plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
#     plt.plot([0, 1], [0, 1], "k--", label="Random")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Overall ROC Curve")
#     plt.legend()
#     plt.grid()

#     # PR
#     plt.subplot(1, 2, 2)
#     plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Overall Precision-Recall Curve")
#     plt.legend()
#     plt.ylim([0.75, 1.])
#     plt.grid()

#     plt.tight_layout()
#     plt.pause(0.5)
#     plt.show(block=False)


def annotate_threshold_markers(
    x_coords,
    y_coords,
    thresholds,
    target_thresholds=[0.1, 0.3, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
):
    """
    Annotate points on a plot for threshold values closest to target_thresholds.

    Args:
        x_coords: x-coordinates for plotting (e.g., fpr or recall)
        y_coords: y-coordinates for plotting (e.g., tpr or precision)
        thresholds: threshold values from ROC or PR curve
        target_thresholds: list of threshold values we want to highlight
    """
    for target in target_thresholds:
        # Find index of closest threshold value
        idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - target))

        # Plot marker and annotate
        plt.plot(x_coords[idx], y_coords[idx], "ro")  # red dot
        plt.text(
            x_coords[idx],
            y_coords[idx],
            f"{thresholds[idx]:.2f}",
            fontsize=8,
            ha="right",
        )


def plot_curves_with_thresholds_and_markers(
    list_of_y_true, list_of_y_scores, split="", confidence="", max_labels=10
):
    # Flatten all true labels and scores
    all_y_true = [label for sublist in list_of_y_true for label in sublist]
    all_y_scores = [score for sublist in list_of_y_scores for score in sublist]

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(all_y_true, all_y_scores)
    roc_auc = roc_auc_score(all_y_true, all_y_scores)

    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(all_y_true, all_y_scores)
    pr_auc = average_precision_score(all_y_true, all_y_scores)

    # Plot
    plt.figure(figsize=(14, 6))

    # ROC plot
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")

    # Annotate thresholds and mark points
    annotate_threshold_markers(fpr, tpr, roc_thresholds)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split}:  ROC Curve{confidence}")
    plt.legend()
    plt.grid()

    # PR plot
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}", lw=2)

    # Annotate thresholds and mark points
    annotate_threshold_markers(recall, precision, pr_thresholds)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split}:  Precision-Recall Curve{confidence}")
    plt.ylim([0.75, 1.02])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.pause(0.5)
    plt.show(block=False)


def process_audio(audio, speech_detector):
    """Returns VAD computed speech probabilities from audio chunks."""
    speech_detector.reset()
    chunk_size = speech_detector.chunk_size
    num_chunks = len(audio) // chunk_size
    return [
        speech_detector(audio[i * chunk_size : (i + 1) * chunk_size])
        for i in range(num_chunks)
    ]


def main():
    dataset = load_dataset("guynich/librispeech_asr_test_vad")

    results = {}
    separator = "=" * 80

    for split in SPLITS:
        all_speech = []
        all_vad_probs = []

        all_speech_confidence = []
        all_vad_probs_confidence = []

        for index, example in enumerate(dataset[split]):
            speech = example["speech"]
            confidence = example["confidence"]
            speech_confidence = np.array(speech)[np.array(confidence) == 1]

            vad_probs = process_audio(example["audio"]["array"], speech_detector)
            vad_probs_confidence = np.array(vad_probs)[np.array(confidence) == 1]

            all_speech.append(speech)
            all_vad_probs.append(vad_probs)

            all_speech_confidence.append(speech_confidence)
            all_vad_probs_confidence.append(vad_probs_confidence)

            # Apply a threshold to vad model probabilities.
            vad_speech = np.array(vad_probs) > VAD_THRESHOLD
            vad_speech_confidence = np.array(vad_probs_confidence) > VAD_THRESHOLD
            metrics = evaluate_binary_classification(speech, vad_speech)
            metrics_confidence = evaluate_binary_classification(
                speech_confidence, vad_speech_confidence
            )

            print(
                f"Example: [{index:04d}]  Split: {split}",  #  Ratio: {ratio:7.2%}  "
            )
            pprint.pprint(metrics)
            pprint.pprint(metrics_confidence)

        results[split] = compute_overall_auc(all_speech, all_vad_probs)
        results[split + "_confidence"] = compute_overall_auc(
            all_speech_confidence, all_vad_probs_confidence
        )
        pprint.pprint(results)
        print(separator)

        # plot_overall_curves(all_speech, all_vad_probs)
        plot_curves_with_thresholds_and_markers(all_speech, all_vad_probs, split=split)
        plot_curves_with_thresholds_and_markers(
            all_speech_confidence,
            all_vad_probs_confidence,
            split=split,
            confidence=" (exclude low confidence)",
        )

    pprint.pprint(results)

    input()


if __name__ == "__main__":
    speech_detector = SpeechDetector()
    main()
