# WGAN-with-gradient-penalty Conditional GAN with WGAN-GP for Sketch-to-Image Generation

## 👨‍💻 Contributors

- **Sahil**  
- **Aman Kanshotia** 
- **Sougata Moi** 
- **Program:** M.Tech & M.Sc — Data and Computational Sciences

---

## 📦 Dataset: ISIC 2016 Skin Lesion Dataset

The dataset includes:

- 🖼️ **Training Images:** 9015
- 🏷️ **Training Labels**
- ✏️ **Training Sketches:** Paired & Unpaired
- 🧪 **Test Images:** 1000
- 🏷️ **Test Labels**
- ✏️ **Test Sketches:** Paired & Unpaired

---

## 🧠 Objective

Develop a **Conditional Generative Adversarial Network (CGAN)** using **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to translate **sketches into realistic skin lesion images**.

---

## 🔬 Methodology

- **Conditional GAN:** Sketch + Label → Image
- **WGAN-GP:** 
  - Enforces **Lipschitz constraint** using gradient penalty
  - Stabilizes training, avoids mode collapse
- **Loss:** Wasserstein distance approximation
- **Evaluation:** Inception Score (IS), Frechet Inception Distance (FID), Classification Accuracy

---

## ⚙️ Experimental Setup

- 🎯 Input Resolution: `64x64`
- 🧠 Conditional GAN Architecture
- 🧮 Optimizer: Adam
- 🔁 Epochs: `100`
- 🔢 Critic Iterations: `5`
- 🔧 Learning Rate: `1e-4`
- λ (Gradient Penalty Coefficient): `100`

---

## 🏗️ Model Architecture

### 🧬 Generator

- Inputs: Sketch + Label
- Label embedded into vector via Embedding layer
- Series of Transposed Convolutional layers + BatchNorm + ReLU
- Final layer uses **tanh** activation
- Upsampling with bilinear interpolation

### 🕵️‍♂️ Discriminator (Critic)

- Inputs: Image + Label
- Label embedded and concatenated with input
- Convolutional layers + InstanceNorm + LeakyReLU
- Final scalar output: authenticity score

## 🧪 Results

### 🎨 Image Generation Samples

#### ✅ Training with Unpaired Sketches
- **Epochs:** 0 → 99  
- Gradual improvement in output image quality  
- Clearer lesion boundaries and skin texture over time  

---

#### ✅ Training with Paired Sketches
- Faster convergence  
- Cleaner and more realistic generations even at early epochs  

---

### 🧪 Testing Phase
- Test sketches translated to images  
- Shows generalization across unseen data  

---

### 📉 Loss Curves
- Generator loss decreases  
- Critic (Discriminator) loss increases  
- Training remains stable throughout  

---

### 📊 Evaluation Metrics

| Metric                          | Dataset             | Value     |
|---------------------------------|---------------------|-----------|
| **Inception Score (No FT)**     | Real Test Images    | 0.0023    |
| **Inception Score (No FT)**     | Generated Images    | 0.0023    |
| **Inception Score (With FT)**   | Real Test Images    | 1.60      |
| **Inception Score (With FT)**   | Generated Images    | 1.43      |
| **FID Score**                   | Generated vs. Real  | 69.9046   |


---

### 🧠 Classification Results (EfficientNet)

Used a **fine-tuned EfficientNet** to classify real vs. generated images.

| Dataset                    | Accuracy (%) |
|----------------------------|--------------|
| Training Data (Real)       | 85.07        |
| Validation Data (Real)     | 74.89        |
| Test Data (Real)           | 63.00        |
| Test Data (Generated)      | 39.41        |

---

## 🔍 Analysis

- WGAN-GP helped mitigate mode collapse and training instability  
- Generated image quality is **better with paired sketches**  
- Inception Scores (IS) were initially low due to **ImageNet–ISIC domain mismatch**  
- After **fine-tuning the classifier**, scores improved significantly  
- The model demonstrated good **generalization** on unseen test sketches

---

## ✅ Conclusion

- ✅ **CGANs** allow for **label-controlled image generation**
- 🧠 **WGAN-GP** improves convergence and training stability
- 📈 **Paired sketches** significantly outperform unpaired ones with sufficient training
- 🔧 **Future improvements**:
  - Longer training duration
  - Hyperparameter tuning
  - Enhanced generator architectures

---

## 📚 Resources

- [Improved Training of WGANs (arXiv:1704.00028)](https://arxiv.org/abs/1704.00028)  
- [WGAN-GP Medium Summary](https://sh-tsang.medium.com/brief-review-wgan-gp-improved-training-of-wasserstein-gans-ae3e2acb25b3)  
- [Wasserstein GAN Paper (arXiv:1701.07875)](https://arxiv.org/abs/1701.07875)  
- Lecture Slides provided by the Instructor
