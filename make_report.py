"""Create the submission report from measured training results: python make_report.py."""
import argparse
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

import report_text as rt
from report_text import draw_architecture_diagram


def main(report_path="CNN_Project_Report.docx"):
    output = Path("results")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    tuning_path = output / "tuning.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8")) if tuning_path.exists() else None
    final, final_label = rt.resolve_final(metrics, tuning)
    draw_architecture_diagram(output / "architecture_diagram.png", dropout=metrics.get("dropout", 0.3))
    doc = Document()
    doc.core_properties.author = "Albert Kabore"
    section = doc.sections[0]
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading("Image Classification with Convolutional Neural Networks", 0)
    doc.add_paragraph("Albert Kabore, PhD student in AI")
    doc.add_paragraph("Neural Networks Implementation Project Report")

    figure_n = 0

    def add_figure(image_name, caption, width):
        nonlocal figure_n
        figure_n += 1
        doc.add_picture(str(output / image_name), width=Inches(width))
        doc.add_paragraph(f"Figure {figure_n}. {caption}")

    def add_table(headers, rows):
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Light Shading Accent 1"
        for cell, title in zip(table.rows[0].cells, headers):
            cell.text = title
        for row in rows:
            for cell, value in zip(table.add_row().cells, row):
                cell.text = str(value)

    # --- Introduction: Problem description and dataset overview ---
    doc.add_heading("Introduction", 1)
    doc.add_paragraph(rt.introduction(metrics))
    add_table(("Class", "Images"), list(metrics["class_counts"].items()))
    add_figure("dataset_overview.png", "Natural Images class counts and one example image from each category. Class frequencies range from 702 dogs to 1,000 fruit images.", 6.8)

    # --- Methodology ---
    doc.add_heading("Methodology", 1)
    doc.add_heading("Preprocessing steps", 2)
    doc.add_paragraph(rt.preprocessing(metrics))
    doc.add_paragraph("Class labels were inferred from the eight folder names. The training loader shuffles images; validation and test images receive deterministic resizing and scaling only. The stratified split preserves the unequal class frequencies across subsets.")

    doc.add_heading("CNN architecture design", 2)
    doc.add_paragraph(rt.architecture(metrics))
    add_figure("architecture_diagram.png", "Shared CNN architecture. The regularized variant additionally applies batch normalization and dropout as noted in each block.", 6.9)

    doc.add_heading("Training setup and hyperparameters", 2)
    doc.add_paragraph(rt.training_setup(metrics))
    add_table(
        ("Setting", "Value"),
        [
            ("Framework / device", f"PyTorch / {metrics['device']}"),
            ("Input", f"RGB, {metrics['image_size']} × {metrics['image_size']} pixels; values 0–1"),
            ("Optimizer / loss", "Adam / cross entropy"),
            ("Initial learning rate", metrics["learning_rate"]),
            ("Batch size / epochs", f"{metrics['batch_size']} / {metrics['epochs_per_variant']} per architecture"),
            ("Regularized variant", f"Batch normalization; {metrics.get('dropout', 0.3):.0%} dropout"),
            ("Random seed / selection", f"{metrics['seed']} / highest validation accuracy"),
        ],
    )
    if tuning:
        doc.add_paragraph(rt.tuning_intro(tuning))

    # --- Results ---
    doc.add_heading("Results", 1)
    doc.add_heading("Performance metrics", 2)
    for line in rt.results_summary_lines(metrics, test):
        doc.add_paragraph(line)
    doc.add_paragraph("Macro metrics average the eight classes equally. The test accuracy uses all held-out examples. Per-class precision, recall, and F1-score are shown below.")
    add_table(
        ("Class", "Precision", "Recall", "F1-score", "Support"),
        [(name, f"{test['classification_report'][name]['precision']:.3f}", f"{test['classification_report'][name]['recall']:.3f}",
          f"{test['classification_report'][name]['f1-score']:.3f}", int(test['classification_report'][name]['support']))
         for name in metrics["class_counts"]],
    )

    doc.add_heading("Confusion matrix and classification report", 2)
    add_figure("confusion_matrix.png", "Confusion matrix for the selected model on the held-out test set. Rows are true labels; columns are predictions.", 5.6)

    doc.add_heading("Training/validation loss and accuracy curves", 2)
    add_figure("training_curves.png", "Training and validation loss and accuracy for both CNN variants. Each dashed line is validation performance.", 6.2)
    baseline = metrics["history"]["baseline"]
    regularized = metrics["history"]["regularized"]
    doc.add_paragraph(
        f"Across three epochs, baseline validation accuracy rose from {baseline['val_accuracy'][0]:.1%} "
        f"to {baseline['val_accuracy'][-1]:.1%}, while the regularized variant rose from "
        f"{regularized['val_accuracy'][0]:.1%} to {regularized['val_accuracy'][-1]:.1%}. "
        "Validation losses declined for both models over the observed training period; "
        "the curves do not establish convergence after only three epochs."
    )
    doc.add_heading("Visual summary", 2)
    add_figure("performance_dashboard.png", "Validation architecture comparison, held-out class metrics, aggregate test metrics, and separate learning-rate and dropout sweeps.", 6.6)

    # --- Discussion ---
    doc.add_heading("Discussion", 1)
    doc.add_heading("Observations, challenges, and insights", 2)
    for paragraph in rt.observations_paragraphs(metrics, final):
        doc.add_paragraph(paragraph)

    doc.add_heading("Justification for design and optimization decisions", 2)
    doc.add_paragraph(rt.justification_intro(metrics))
    if tuning:
        add_table(
            ("Learning rate", "Best validation accuracy"),
            [(lr, f"{tuning['learning_rate_sweep'][str(lr)]:.3%}") for lr in tuning["learning_rates_tried"]],
        )
        doc.add_paragraph(f"Best learning rate: {tuning['best_learning_rate']}.")
        add_table(
            ("Dropout rate", "Best validation accuracy"),
            [(rate, f"{tuning['dropout_sweep_at_best_lr'][str(rate)]:.3%}") for rate in tuning["dropout_rates_tried"]],
        )
        doc.add_paragraph(f"Best dropout rate at learning rate {tuning['best_learning_rate']}: {tuning['best_dropout']}.")
        add_figure("tuning_curves.png", "Best validation accuracy across the learning rate sweep (left) and the dropout sweep at the best learning rate (right).", 6.2)
        doc.add_paragraph(rt.tuning_outcome(metrics, tuning, final))
        note = rt.tuning_variance_note(tuning, metrics)
        if note:
            doc.add_paragraph(note)

    doc.add_heading("Limitations", 2)
    doc.add_paragraph(
        "The dataset was evaluated with one stratified split and one initial seed. Three-epoch "
        "runs and a two-stage search give a useful comparison but do not measure uncertainty "
        "across seeds or establish convergence. Batch normalization and dropout were changed "
        "together, so their individual effects cannot be inferred. The initial test result was "
        "viewed before the tuning sweep; although tuning selected configurations by validation "
        "accuracy, a fresh test set would provide a stronger final unbiased estimate. The test "
        "set describes performance on this dataset; robustness to new sources, lighting, viewpoints, and "
        "class definitions remains unmeasured. Dog recall of 0.171 is a material limitation "
        "despite the overall 0.746 accuracy."
    )

    doc.add_heading("Potential improvements and future work", 2)
    doc.add_paragraph(rt.future_work())

    # --- Final insight and recommendations ---
    doc.add_heading("Final Insight and Recommendations", 1)
    for paragraph in rt.final_insight_and_recommendations(metrics, tuning, final, final_label):
        doc.add_paragraph(paragraph)

    doc.add_heading("Reproducibility and source", 1)
    doc.add_paragraph(rt.reproducibility(metrics))

    # --- Conclusion: always last ---
    doc.add_heading("Conclusion", 1)
    doc.add_paragraph(rt.conclusion(metrics, tuning, final, final_label))

    doc.save(report_path)
    print(f"Created {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="CNN_Project_Report.docx")
    main(parser.parse_args().output)
