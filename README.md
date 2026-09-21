# Image Classification with Convolutional Neural Networks

This project trains and compares two PyTorch CNNs on the eight-class [Natural Images dataset](https://www.kaggle.com/datasets/prasunroy/natural-images). It uses a stratified 70/15/15 train/validation/test split and reports held-out accuracy, macro precision, macro recall, macro F1, a classification report, and a confusion matrix. The winning architecture is then tuned over learning rate and dropout rate.

The submission notebook is `Image_Classification_CNN.ipynb`. Run its cells in order after installing the requirements. The notebook contains the complete training and tuning implementation, plus a full Introduction/Methodology/Results/Optimization/Discussion/Conclusion write-up rendered from the measured results at the end; `train.py` and `tune.py` offer the same workflow from the command line, and `report_text.py` holds the narrative text and architecture diagram shared by the notebook and `CNN_Project_Report.docx`.

## Run

```powershell
python -m pip install -r requirements.txt
python train.py --data "C:\path\to\natural_images" --epochs 3
python tune.py
python make_report.py
```

The dataset directory must contain `airplane`, `car`, `cat`, `dog`, `flower`, `fruit`, `motorbike`, and `person` folders. On this machine, `train.py` defaults to the local dataset path; use `--data` elsewhere. Results are saved in `results/`; the report is `CNN_Project_Report.docx`. The fixed seed is 42. CPU training time depends on the machine.

The baseline CNN and the batch-normalization/dropout CNN share the same three convolution blocks. Best checkpoints are chosen by validation accuracy. Only the selected checkpoint is evaluated on the test set.

`tune.py` takes the architecture that won the comparison above and searches learning rate, then dropout rate, at 3 epochs per configuration, using the same validation split. If the tuned configuration beats the original's validation accuracy, it is re-evaluated on the test set once and becomes the final model; otherwise the original selected model stands. Results are saved to `results/tuning.json` and `results/tuning_curves.png`.
