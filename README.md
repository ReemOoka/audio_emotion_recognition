# 🎙️ Audio Emotion Recognizer Research Project

## Google Colab Notebooks

- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10KJNwR9PpfmUY5s4pIdLKwcyUfF1vdUM?usp=sharing)  
  **Audio Emotion Recognition + SER.ipynb**  

- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WF_-mt-7vC9dTpyXaGGClFoa1cjwHLz6?usp=sharing)  
  **Log-Mel CNN model.ipynb**  

An end-to-end AI/ML pipeline for detecting emotions in speech, built from scratch as part of my research assistant role at Marist University. While existing systems rely on off-the-shelf emotion API’s, I implemented three complementary models to explore classical audio features, NLP transformers, and direct spectrogram image learning—all trained and evaluated on five benchmark datasets.

---

## 🚀 Motivation

Human communication is driven as much by **tone**, **pitch**, and **rhythm** as by words. Machines today struggle to “hear” feelings. My goal: build a system that  
1. **Listens** to an audio clip  
2. **Analyzes** both raw sound and transcribed speech  
3. **Predicts** the speaker’s emotional state across multiple classes  

Use cases span **empathetic chatbots**, **mental-health monitoring**, **gaming/VR**, **e-learning**, and **call-center analytics**.

---

## 🏗️ Models Overview

| Model   | Approach                                                            | Data         | Accuracy  |
|:-------:|:--------------------------------------------------------------------|:------------:|:---------:|
| **A**   | 1D-CNN on audio features (MFCC, ZCR, RMS)                           | 12 000+ clips| ~98%      |
| **B**   | Speech → transcript → DistilRoBERTa transformer                     | 12 000+ clips| 95%+      |
| **C**   | 2D-CNN on 128×128 log-Mel spectrograms                              | 600 clips    | ~93%      |

1. **Model A** uses low-level **signal features** and a deep 1D-CNN for fast, interpretable classification.  
2. **Model B** leverages **Hugging Face**’s transformers to infer emotion from transcribed text.  
3. **Model C** treats spectrograms as images and trains a compact 2D-CNN end-to-end—no manual feature engineering required.

---

## 🗂️ Datasets

I combined **five** public benchmarks:

| Dataset  | Content                                  | Samples/Size           |
|:--------:|:-----------------------------------------|:-----------------------|
| RAVDESS  | 24 actors, 8 emotions, 2 intensities     | 1 440 clips            |
| SAVEE    | 4 male actors, 7 emotions                | 480 utterances         |
| TESS     | 2 actresses, 7 emotions                  | 2 800 clips            |
| CREMA-D  | 91 actors, 6 emotions                    | 7 442 audio samples    |
| IEMOCAP  | 10 actors in dialogues, 5 emotions       | ~12 hrs recordings     |

**Final training set** for Model C: 4 emotions (happy, sad, angry, neutral), 150 samples/emotion ⇒ 600 clean spectrograms.

---

## 📊 Key Results
Model	Validation Accuracy	Notes
A	98%	12 000+ samples, deep 1D-CNN on MFCC/ZCR/RMS
B	95%+	Transformer NLP on speech transcripts
C	93%	600 clean spectrograms, 2D-CNN, no augmentation

### 🎥 Live Demo and Graphs and Matrix

<img width="815" alt="image" src="https://github.com/user-attachments/assets/e68230e8-8df0-4c9d-82b6-8c9d9908fd3c" />

<img width="842" alt="image" src="https://github.com/user-attachments/assets/b69023fc-4dd1-4b78-9a14-1267f4ea8d80" />

<img width="857" alt="image" src="https://github.com/user-attachments/assets/9f293dcd-e5f4-40e2-9f21-d14a1e6c7a0c" />

<img width="859" alt="image" src="https://github.com/user-attachments/assets/0d6b0e09-e91c-413c-820f-400f54b6f54f" />

<img width="887" alt="image" src="https://github.com/user-attachments/assets/91293081-7dfa-4fd4-bc98-81e5ed727733" />

![image](https://github.com/user-attachments/assets/29051cab-b5f0-492f-a5d7-e2dca845b59d)

![image](https://github.com/user-attachments/assets/5e3aa4b1-efdb-418c-861e-015e6e5bcdd0)

<img width="335" alt="image" src="https://github.com/user-attachments/assets/b387229a-7c57-418f-8f11-baf4a6e42200" />

![image](https://github.com/user-attachments/assets/929db5ac-d3c1-429b-aef2-efce8ffa121f)

![image](https://github.com/user-attachments/assets/dd46e9d2-f5d6-4fc4-8b67-66904094dc05)

![image](https://github.com/user-attachments/assets/594489b4-be49-41fd-8d49-a3b9add42c65)

![image](https://github.com/user-attachments/assets/46f22b2e-5823-4ff5-b1f4-cc46c58f188c)

![image](https://github.com/user-attachments/assets/80fa763e-eb42-4382-9f7e-69d144d85649)

<img width="875" alt="image" src="https://github.com/user-attachments/assets/f860a538-c74b-4106-ac1c-b8d6a5f7561a" />

## 📓 Colab Notebooks

You can explore the full end-to-end pipeline in Google Colab:

- **Audio Features & Speech Recognition**  
  - Open directly:  
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10KJNwR9PpfmUY5s4pIdLKwcyUfF1vdUM?usp=sharing)  
  - Or download `Audio Emotion Recognition + SER.ipynb` from this repo’s `notebooks/` folder and upload to Colab.

- **Log-Mel Spectrogram CNN & Comparison**  
  - Open directly:  
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WF_-mt-7vC9dTpyXaGGClFoa1cjwHLz6?usp=sharing)  
  - Or download `Log-Mel CNN model.ipynb` from `notebooks/` and run locally or in Colab.

> **Note:** All other assets—feature Parquet files, pretrained models (`.h5`, `.pkl`), scalers, and encoders—are available under the [Releases] section.  

---


## 💾 Download Data & Models
Use my GitHub Releases (v1–v6) to fetch Parquet files, pretrained weights, encoders, model and scalers:

### 🛠️ Installation & Quick Start
 ```bash
git clone https://github.com/ReemOoka/audio-emotion-recognition.git
cd audio-emotion-recognition
 ```

---

## 🔍 Evaluation & Discussion
Model A converges in <10 epochs, with <5% train/val gap by epoch 5.

Model C shows balanced recall (>90%) across all four emotions.

Real-world robustness: clean-audio demo achieves 93% accuracy on 2.5–3 s clips; live-record mode supports smart segmentation for longer files.

## 🌟 Future Work
Advanced augmentation: targeted noise, enviromental sounds,etc.

Multimodal fusion: combine audio, text, and facial cues

Real-time streaming: continuous emotion tracking in live audio

Multilingual support: extend to diverse languages & accents

Model compression: quantization/pruning for edge deployment

## ❓ Questions for Review
Why did naïve noise/pitch/time augmentations fail to boost Model C’s performance?

Which preprocessing steps (voice activity detection, dynamic range compression) could improve accuracy?

How would you integrate pretrained audio encoders (e.g., wav2vec2) as feature backbones?

What metrics & monitoring practices do you recommend for model drift in production?

Have you found any best practices for balancing augmentation diversity vs. data quality?

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Please open an issue or submit a PR.

## 📝 License
This project is licensed under the  Apache License Version 2.0. See LICENSE for details.

## Author
Reem Ooka — MS ‘25, Marist University (AI/ML Research Assistant)
