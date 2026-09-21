"""Narrative text and figures shared by the .docx report and the notebook, built from measured results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_architecture_diagram(path, dropout=0.3):
    labels = [
        "Input\n48x48x3",
        "Conv 3x3, 16\n+BN if regularized\nReLU, MaxPool 2x2",
        "Conv 3x3, 32\n+BN if regularized\nReLU, MaxPool 2x2",
        "Conv 3x3, 64\n+BN if regularized\nReLU, MaxPool 2x2",
        "Global\nAvg Pool",
        f"Dropout {dropout}\nif regularized",
        "Linear\n-> 8 classes",
    ]
    box_w, box_h, gap = 1.9, 1.0, 0.45
    fig_w = len(labels) * box_w + (len(labels) - 1) * gap + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, 2.2))
    ax.axis("off")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, box_h + 0.6)
    x, rights = 0.3, []
    for label in labels:
        ax.add_patch(FancyBboxPatch((x, 0.3), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     linewidth=1.3, edgecolor="#2b6cb0", facecolor="#ebf4ff"))
        ax.text(x + box_w / 2, 0.3 + box_h / 2, label, ha="center", va="center", fontsize=8.5)
        rights.append(x + box_w)
        x += box_w + gap
    for right in rights[:-1]:
        ax.annotate("", xy=(right + gap, 0.3 + box_h / 2), xytext=(right, 0.3 + box_h / 2),
                     arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=1.3))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def introduction(metrics):
    total = sum(metrics["class_counts"].values())
    return (f"This project studies eight-class image classification with PyTorch CNNs. "
            f"The Natural Images dataset contains {total:,} labeled photographs of airplanes, "
            "cars, cats, dogs, flowers, fruit, motorbikes, and people. The question is whether "
            "adding batch normalization and dropout to a compact CNN improves classification "
            "when both variants use the same data split, depth, width, optimizer, and training "
            "schedule. Model selection uses validation accuracy; accuracy and class-sensitive "
            "metrics on held-out images describe the final model's performance.")


def preprocessing(metrics):
    split = metrics["split"]
    return (f"Images were divided with stratified random sampling (seed {metrics['seed']}) into "
            f"{split['train']} training, {split['validation']} validation, and {split['test']} "
            "test examples (70/15/15). Validation accuracy selected the architecture; its "
            "checkpoint was then evaluated on the test set before the later tuning sweep. "
            f"All images were converted to RGB, resized to {metrics['image_size']} x "
            f"{metrics['image_size']} pixels, and scaled from 0-255 to 0-1. Training images alone "
            "received random horizontal flips and brightness changes (factor 0.85-1.15).")


def architecture(metrics):
    dropout = metrics.get("dropout", 0.3)
    return ("Both networks use three 3x3 convolution blocks with 16, 32, and 64 output channels. "
            "Each block has ReLU and 2x2 max pooling. Global average pooling maps each channel to "
            "one value, followed by a linear eight-class classifier. The regularized variant adds "
            f"batch normalization after every convolution and {dropout:.0%} dropout before the "
            "classifier. The baseline omits both. This comparison tests the combined design "
            "change while keeping network width and depth fixed.")


def training_setup(metrics):
    return (f"Both variants were trained for {metrics['epochs_per_variant']} epochs using cross "
            f"entropy loss, Adam with learning rate {metrics['learning_rate']}, and batch size "
            f"{metrics['batch_size']}. The checkpoint with the highest validation accuracy for "
            "each variant was saved; the variant with the higher validation accuracy was "
            f"selected. Ties favor the baseline. Training ran on {metrics['device']}. No test "
            "labels were used for optimization. These starting hyperparameters were then tuned "
            "by a grid search over learning rate and dropout rate, reported and justified in the "
            "Discussion section.")


def results_summary_lines(metrics, test):
    lines = [f"{name.capitalize()} best validation accuracy: {score:.3%}" for name, score in metrics["validation_accuracy"].items()]
    lines.append(
        f"Selected model: {metrics['selected_model']}. Test accuracy: {test['accuracy']:.3%}; "
        f"macro precision: {test['macro_precision']:.3f}; macro recall: {test['macro_recall']:.3f}; "
        f"macro F1: {test['macro_f1']:.3f}; test cross entropy: {test['loss']:.3f}."
    )
    return lines


def tuning_intro(tuning):
    return (f"Starting from the winning architecture, learning rate and dropout were tuned by a "
            f"two-stage grid search, each run for {tuning['epochs_per_run']} epochs and scored by "
            "validation accuracy. First, the learning rate was swept with dropout fixed at its "
            "original value; then dropout was swept at the best learning rate found. This keeps "
            "the search small (six training runs, including one repeated configuration) while "
            "probing both an optimization and a regularization hyperparameter.")


def tuning_outcome(metrics, tuning, final):
    if tuning["improved_over_prior"]:
        return (f"The tuned configuration (learning rate {tuning['best_learning_rate']}, dropout "
                f"{tuning['best_dropout']}) reached {tuning['tuned_validation_accuracy']:.3%} "
                f"validation accuracy, exceeding the original {tuning['prior_validation_accuracy']:.3%}. "
                "It was therefore re-evaluated once on the held-out test set as the final model, "
                f"reaching {final['accuracy']:.3%} accuracy and {final['macro_f1']:.3f} macro F1.")
    dropout = metrics.get("dropout", 0.3)
    return (f"The best dropout-sweep run reached {tuning['tuned_validation_accuracy']:.3%} "
            f"validation accuracy, which did not exceed the original {tuning['prior_validation_accuracy']:.3%} "
            f"achieved with learning rate {metrics['learning_rate']} and dropout {dropout}. The learning-rate "
            "sweep matched the original score at the same settings; the repeated dropout-sweep "
            "run scored lower. The original "
            f"{metrics['selected_model']} model was therefore kept as the final model, and its test "
            "results above are unchanged.")


def tuning_variance_note(tuning, metrics):
    lr_table_best = max(tuning["learning_rate_sweep"].values())
    if lr_table_best > tuning["tuned_validation_accuracy"] + 0.02:
        dropout = metrics.get("dropout", 0.3)
        val_size = metrics["split"]["validation"]
        return (f"Note the learning rate sweep's top result ({lr_table_best:.3%}) was higher than "
                "any dropout-sweep run at that same learning rate, even the one re-run at the "
                f"original dropout of {dropout}. Each configuration here trained for only "
                f"{tuning['epochs_per_run']} epochs on a {val_size}-image validation set with random "
                "augmentation, so a single run's best-epoch accuracy is noisy; that top score is "
                "best read as an optimistic outlier rather than a reproducible estimate. A more "
                "conservative search would average several seeds per configuration before "
                "comparing them.")
    return None


def confusion_insight(metrics, final, top_n=3):
    classes = list(metrics["class_counts"].keys())
    matrix = final["confusion_matrix"]
    mistakes = sorted(
        ((matrix[i][j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and matrix[i][j] > 0),
        reverse=True,
    )
    if not mistakes:
        return "every test image was classified correctly, so the confusion matrix has no off-diagonal entries."
    parts = [f"{count} true '{true_class}' images predicted as '{predicted_class}'" for count, true_class, predicted_class in mistakes[:top_n]]
    return "the largest confusion errors on the test set were " + "; ".join(parts) + "."


def observations_paragraphs(metrics, final):
    class_rows = [(name, final["classification_report"][name]["recall"]) for name in metrics["class_counts"]]
    weakest = min(class_rows, key=lambda item: item[1])
    strongest = max(class_rows, key=lambda item: item[1])
    p1 = (f"On the test set, recall was highest for {strongest[0]} ({strongest[1]:.3f}) and "
          f"lowest for {weakest[0]} ({weakest[1]:.3f}). Per the confusion matrix, "
          f"{confusion_insight(metrics, final)} This shows that aggregate accuracy alone does not "
          "describe performance across all categories, and that per-class recall is a more "
          "informative diagnostic than a single accuracy number.")
    p2 = (f"Training used a CPU, so images were resized to {metrics['image_size']}x"
          f"{metrics['image_size']} and the CNN was limited to three convolution blocks. "
          f"With {metrics['epochs_per_variant']} epochs per run, neither learning curve proves "
          "convergence. The baseline's lower training and validation accuracy suggests that "
          "optimization and model capacity, as well as regularization, could affect the observed "
          "gap. The repeated learning-rate/dropout setting produced different validation scores "
          "in the two search stages; this illustrates uncertainty from short stochastic runs, "
          "but two observations are insufficient to estimate run-to-run variance reliably.")
    return [p1, p2]


def justification_intro(metrics):
    base = metrics["validation_accuracy"].get("baseline")
    reg = metrics["validation_accuracy"].get("regularized")
    dropout = metrics.get("dropout", 0.3)
    return (f"The combined batch-normalization and {dropout:.0%}-dropout architecture was "
            f"selected over the baseline because it reached {reg:.3%} best validation accuracy versus "
            f"{base:.3%} for the baseline under identical data, epochs, and optimizer settings. "
            "Batch normalization normalizes minibatch activations; dropout randomly removes "
            "classifier inputs during training. Because both were introduced together, these "
            "experiments do not isolate either component's individual contribution. Global "
            "average pooling limits classifier parameters compared with flattening a full "
            "feature map, which suits the small training set and CPU budget.")


def future_work():
    return ("Future work could compare larger input sizes, more training epochs with early "
            "stopping, class-weighted loss to offset the uneven class counts, additional "
            "augmentation such as rotation and cropping, and pretrained feature extractors "
            "(transfer learning) rather than a network trained from scratch. Any of these changes "
            "would need a new validation comparison and a fresh held-out test set for a fair final "
            "estimate. Averaging several random seeds per hyperparameter configuration would also "
            "make the tuning comparison more reliable; the repeated three-epoch configuration "
            "produced different validation scores.")


def final_insight_and_recommendations(metrics, tuning, final, final_label):
    base = metrics["validation_accuracy"].get("baseline")
    reg = metrics["validation_accuracy"].get("regularized")
    class_rows = [(name, final["classification_report"][name]["recall"], final["classification_report"][name]["f1-score"])
                  for name in metrics["class_counts"]]
    weakest = min(class_rows, key=lambda item: item[1])
    lines = [
        f"The largest measured change in this comparison came from the combined architecture "
        f"modification: adding batch normalization and dropout raised best validation "
        f"accuracy by {(reg - base) * 100:.1f} percentage points (from {base:.3%} to {reg:.3%}), a larger gain than any "
        "improvement retained after the learning-rate and dropout search. This comparison "
        "does not separate the effects of batch normalization and dropout."
    ]
    if tuning:
        gap = tuning["tuned_validation_accuracy"] - tuning["prior_validation_accuracy"]
        if tuning["improved_over_prior"]:
            lines.append(
                f"The hyperparameter search added a further {gap:.1%} on top of that, moving best "
                f"validation accuracy to {tuning['tuned_validation_accuracy']:.3%}."
            )
        else:
            lines.append(
                f"The hyperparameter search did not improve on the original settings (best "
                f"dropout-sweep run {tuning['tuned_validation_accuracy']:.3%} versus "
                f"{tuning['prior_validation_accuracy']:.3%} original). Within the values and "
                "three-epoch runs tested, no further gain was observed from changing learning "
                "rate or dropout alone."
            )
    lines.append(
        f"{weakest[0].capitalize()} is the model's weakest category (recall {weakest[1]:.3f}, "
        f"F1 {weakest[2]:.3f}); {confusion_insight(metrics, final)} Recommendation: prioritize "
        f"more or better-augmented {weakest[0]} training images, or a class-weighted loss, before "
        "any further architecture or hyperparameter changes, since that class has the largest "
        "single share of the observed test errors."
    )
    lines.append(
        f"Recommendation for deployment: the {final_label} model (test accuracy "
        f"{final['accuracy']:.3%}, macro F1 {final['macro_f1']:.3f}) is the strongest checkpoint "
        "measured in this project. These experiments do not establish deployment readiness. "
        f"The low {weakest[0]} recall calls for targeted error analysis and evaluation on new "
        "images before any practical use."
    )
    return lines


def conclusion(metrics, tuning, final, final_label):
    tuning_clause = " and a follow-up hyperparameter search" if tuning else ""
    split = metrics["split"]
    return (f"A reproducible CNN pipeline was implemented for eight-class Natural Images "
            f"classification. The {final_label} CNN was selected from a controlled architecture "
            f"comparison{tuning_clause} and achieved {final['accuracy']:.3%} accuracy and "
            f"{final['macro_f1']:.3f} macro F1 on {split['test']} held-out images. The saved "
            "metrics, curves, confusion matrix, and model checkpoints support reproduction and "
            "further tuning.")


def reproducibility(metrics):
    return ("Dataset: Prasun Roy, Natural Images, Kaggle, "
            "https://www.kaggle.com/datasets/prasunroy/natural-images. The submission notebook "
            "Image_Classification_CNN.ipynb contains data loading, both CNNs, training, "
            "evaluation, tuning, and visual analysis. Set DEFAULT_DATA to the directory "
            "containing the eight class folders, install requirements.txt, and run the notebook "
            f"in order with {metrics['epochs_per_variant']} epochs per architecture. The "
            "stratified split and model initialization use seed 42; augmentation and CPU "
            "execution can still yield small differences across reruns. Metrics and figures "
            "used in this report are preserved in results/.")


def resolve_final(metrics, tuning):
    test = metrics["test"]
    if tuning and tuning.get("improved_over_prior"):
        return tuning["tuned_test"], "tuned regularized"
    return test, metrics["selected_model"]
