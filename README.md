# Real-time Football Player Tracking

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 1. Project Overview

This project is an AI system designed for **real-time football player detection and jersey color & number visibility classification**.  
It consists of two main tasks:

1. **Player Detection** → Detecting the position of players in video frames (bounding boxes).  
2. **Jersey Color & Number Visibility Classification**.

The pipeline is designed to work in **real-time (or near real-time)**, making it suitable for analyzing football matches from live streams or recorded videos.
![Demo animation](demo_output.gif)
---

## 2. Project Structure

```
.
├── config/
│   ├── outputs/ (to keep models (.pth, .pt) files)
│   └── .yaml files (for configuration)
├── data/
│   ├── football_train/
│   ├── football_val/
│   └── football_test/
├── src/
│   ├── dataset/
│   ├── inference/
│   └── trainer/
├── train_detection.sh
├── train_number_visibility.sh
├── train_color.sh
├── requirements.txt
├── data.sh
└── README.md
````

---

## 3. Requirements

- Python **3.10+**  
- **GPU** recommended (CUDA / cuDNN) for training and faster inference  

Install dependencies:

```bash
pip install -r requirements.txt
````



---

## 4. Dataset Preparation

1. Collect match videos/images and annotate exactly by the format of dummy data pushed.
2. Split dataset into `football_train/football_val/football_test`.
3. Run the dataset preparation script (if available):

```bash
bash data.sh
```


---

## 5. Training

### 5.1 Player Detection

Train the detection model:

```bash
bash train_detection.sh
```

### 5.2 Jersey Number Classification

Train classifier models:

```bash
bash train_number_visibility.sh
bash train_color.sh
```

---

## 6. Inference (Real Video)

Run inference on a football match video:

```bash
python src/inference/run_inference.py \
  --model_detection path/to/detection_checkpoint.pth \
  --model_classifier path/to/classifier_checkpoint.pth \
  --input_video path/to/input_video.mp4 \
  --output_video path/to/output_with_predictions.mp4
```

Example shell script:

```bash
#!/bin/bash
python src/inference/infer_player_number.py \
  --det_checkpoint models/detector_best.pth \
  --cls_checkpoint models/number_cls_best.pth \
  --video_input data/videos/match1.mp4 \
  --video_output results/output1.mp4
```

---

## 7. To-do Works

* **Docker**: Adding the feature for the project to run on Docker for multi-platform training.
* **Training**: Doing more experiments to make better models on both detection and classification tasks.


---


## 8. Special Thanks

Special thanks to my teacher **Viet Nguyen** and my mentor **Tien Dang** for instructing & advising me to complete this project.


---

## 9. License

This project is released under the [MIT License](LICENSE).

---


