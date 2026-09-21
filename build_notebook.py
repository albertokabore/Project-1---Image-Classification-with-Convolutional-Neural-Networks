"""Build the self-contained submission notebook from train.py, tune.py, and report_text.py."""
from pathlib import Path

import nbformat as nbf


def strip_main_guard(text):
    return text.split('if __name__ == "__main__":')[0]


train_source = strip_main_guard(Path("train.py").read_text(encoding="utf-8"))
tune_source = strip_main_guard(Path("tune.py").read_text(encoding="utf-8"))
tune_source = tune_source.replace("from train import CLASSES, CNN, DEFAULT_DATA, Images, collect, evaluate, train_variant\n", "")
report_text_source = Path("report_text.py").read_text(encoding="utf-8")

notebook = nbf.v4.new_notebook()
notebook.cells = [
    nbf.v4.new_markdown_cell(
        "# Image Classification with Convolutional Neural Networks\n\n"
        "**Introduction.** This notebook classifies photographs into eight object "
        "categories from the [Natural Images dataset]"
        "(https://www.kaggle.com/datasets/prasunroy/natural-images) using a "
        "convolutional neural network (CNN) built in PyTorch. It covers, in order:\n\n"
        "1. Data collection and preprocessing (loading, normalizing, resizing, augmenting)\n"
        "2. CNN architecture design (a baseline CNN vs. a batch normalization + dropout CNN)\n"
        "3. The training loop and loss function\n"
        "4. Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)\n"
        "5. Hyperparameter tuning/experiments (learning rate and dropout search)\n"
        "6. A full write-up of methodology, results, discussion, conclusions, and final "
        "insight/recommendations, rendered from the measured results below\n\n"
        "Run the cells in order. Training runs on CPU by default and takes a few minutes."
    ),
    nbf.v4.new_markdown_cell("## 1-4. Data loading, preprocessing, CNN architectures, training loop, and evaluation"),
    nbf.v4.new_code_cell(train_source),
    nbf.v4.new_markdown_cell("### Run the architecture comparison"),
    nbf.v4.new_code_cell('import sys\nsys.argv = ["train.py", "--epochs", "3"]\nmain()'),
    nbf.v4.new_markdown_cell(
        "## 5. Hyperparameter tuning/experiments\n\n"
        "Starting from the architecture that won the comparison above, this sweeps the "
        "learning rate, then the dropout rate, at three epochs per configuration, and "
        "promotes the tuned configuration to the final model only if it beats the "
        "original on validation accuracy."
    ),
    nbf.v4.new_code_cell(tune_source),
    nbf.v4.new_code_cell("main()"),
    nbf.v4.new_markdown_cell("## 6. Report: Introduction, Methodology, Results, Discussion, and Conclusion"),
    nbf.v4.new_code_cell(report_text_source),
    nbf.v4.new_code_cell(
        'import json\n'
        'from pathlib import Path\n'
        'from IPython.display import display, Image as DisplayImage, Markdown\n'
        '\n'
        'metrics = json.loads(Path("results/metrics.json").read_text())\n'
        'test = metrics["test"]\n'
        'tuning_path = Path("results/tuning.json")\n'
        'tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else None\n'
        'final, final_label = resolve_final(metrics, tuning)\n'
        '\n'
        'display(Markdown("## Introduction"))\n'
        'display(Markdown(introduction(metrics)))\n'
        'display(Markdown(\n'
        '    "| Class | Images |\\n|---|---|\\n" +\n'
        '    "\\n".join(f"| {name} | {count} |" for name, count in metrics["class_counts"].items())\n'
        '))\n'
        '\n'
        'display(Markdown("## Methodology"))\n'
        'display(Markdown("### Preprocessing steps"))\n'
        'display(Markdown(preprocessing(metrics)))\n'
        'display(Markdown("### CNN architecture design"))\n'
        'display(Markdown(architecture(metrics)))\n'
        'architecture_diagram_path = Path("results/architecture_diagram.png")\n'
        'draw_architecture_diagram(architecture_diagram_path, dropout=metrics.get("dropout", 0.3))\n'
        'display(DisplayImage(filename=str(architecture_diagram_path)))\n'
        'display(Markdown("### Training setup and hyperparameters"))\n'
        'display(Markdown(training_setup(metrics)))\n'
        '\n'
        'display(Markdown("## Results"))\n'
        'display(Markdown("### Performance metrics"))\n'
        'for line in results_summary_lines(metrics, test):\n'
        '    display(Markdown(line))\n'
        'header = "| Class | Precision | Recall | F1-score | Support |\\n|---|---|---|---|---|"\n'
        'rows = [\n'
        '    f"| {name} | {test[\'classification_report\'][name][\'precision\']:.3f} | "\n'
        '    f"{test[\'classification_report\'][name][\'recall\']:.3f} | "\n'
        '    f"{test[\'classification_report\'][name][\'f1-score\']:.3f} | "\n'
        '    f"{int(test[\'classification_report\'][name][\'support\'])} |"\n'
        '    for name in metrics["class_counts"]\n'
        ']\n'
        'display(Markdown(header + "\\n" + "\\n".join(rows)))\n'
        'display(Markdown("### Confusion matrix and classification report"))\n'
        'display(DisplayImage(filename="results/confusion_matrix.png"))\n'
        'display(Markdown("Figure. Confusion matrix for the selected model on the held-out test set."))\n'
        'display(Markdown("### Training/validation loss and accuracy curves"))\n'
        'display(DisplayImage(filename="results/training_curves.png"))\n'
        'display(Markdown("Figure. Training and validation loss and accuracy for both CNN variants."))\n'
        '\n'
        'display(Markdown("## Discussion"))\n'
        'display(Markdown("### Observations, challenges, and insights"))\n'
        'for paragraph in observations_paragraphs(metrics, final):\n'
        '    display(Markdown(paragraph))\n'
        '\n'
        'display(Markdown("### Justification for design and optimization decisions"))\n'
        'display(Markdown(justification_intro(metrics)))\n'
        'if tuning:\n'
        '    display(Markdown(tuning_intro(tuning)))\n'
        '    lr_rows = "\\n".join(f"| {lr} | {tuning[\'learning_rate_sweep\'][str(lr)]:.3%} |" for lr in tuning["learning_rates_tried"])\n'
        '    display(Markdown("| Learning rate | Best val accuracy |\\n|---|---|\\n" + lr_rows))\n'
        '    dropout_rows = "\\n".join(f"| {d} | {tuning[\'dropout_sweep_at_best_lr\'][str(d)]:.3%} |" for d in tuning["dropout_rates_tried"])\n'
        '    display(Markdown("| Dropout rate | Best val accuracy |\\n|---|---|\\n" + dropout_rows))\n'
        '    display(DisplayImage(filename="results/tuning_curves.png"))\n'
        '    display(Markdown(tuning_outcome(metrics, tuning, final)))\n'
        '    note = tuning_variance_note(tuning, metrics)\n'
        '    if note:\n'
        '        display(Markdown(note))\n'
        '\n'
        'display(Markdown("### Potential improvements and future work"))\n'
        'display(Markdown(future_work()))\n'
        '\n'
        'display(Markdown("## Final Insight and Recommendations"))\n'
        'for paragraph in final_insight_and_recommendations(metrics, tuning, final, final_label):\n'
        '    display(Markdown(paragraph))\n'
        '\n'
        'display(Markdown("### Reproducibility and source"))\n'
        'display(Markdown(reproducibility(metrics)))\n'
        '\n'
        'display(Markdown("## Conclusion"))\n'
        'display(Markdown(conclusion(metrics, tuning, final, final_label)))\n'
        '\n'
        'import make_report\n'
        'make_report.main()\n'
        'print("Created CNN_Project_Report.docx")'
    ),
]
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
notebook_path = Path("Image_Classification_CNN.ipynb")
if notebook_path.exists():
    print(f"Kept existing {notebook_path} unchanged")
else:
    nbf.write(notebook, notebook_path)
    print(f"Created {notebook_path}")
