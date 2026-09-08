# VQ‑VAE for Retinopathy of Prematurity (ROP) Fundus Imaging

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white">
  <img alt="Colab" src="https://img.shields.io/badge/Runtime-Colab%20T4%20GPU-F9AB00?style=flat&logo=googlecolab&logoColor=white">
  <img alt="Domain" src="https://img.shields.io/badge/Domain-Medical%20Imaging-0A7E8C?style=flat">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%20prototype-6E56CF?style=flat">
</p>

<p align="center">
  <b>A VQGAN‑style discrete autoencoder that compresses 128×128 RGB retinal fundus images of preterm infants into a 32×32 grid of 20 learned codebook tokens, trained with reconstruction, perceptual, commitment and adversarial objectives.</b>
</p>

<p align="center">
  <code>2.48M params</code> · <code>2,627 images</code> · <code>46 epochs</code> · <code>val recon MSE 0.0052</code> · <code>FID₆₄ 0.18</code>
</p>

***

## Table of contents

1. [Overview](#1-overview)
2. [Headline results](#2-headline-results)
3. [Repository contents](#3-repository-contents)
4. [Method](#4-method)
5. [Results in detail](#5-results-in-detail)
6. [Vessel segmentation stage](#6-vessel-segmentation-stage)
7. [Reproducing the run](#7-reproducing-the-run)
8. [Full configuration reference](#8-full-configuration-reference)
9. [Known limitations](#9-known-limitations)
10. [Roadmap](#10-roadmap)
11. [Suggested layout for a refactor](#11-suggested-layout-for-a-refactor)
12. [References](#12-references)

***

## 1. Overview

Retinopathy of Prematurity is a vasoproliferative disease of the developing retina and one of the leading preventable causes of childhood blindness. Screening produces large volumes of wide‑field fundus photographs, but labelled data is scarce, heavily imbalanced across disease stages, and hard to share across institutions.

This project builds the **compression backbone** that such a pipeline needs: a Vector‑Quantized Variational Autoencoder (VQ‑VAE) with a VQGAN‑style training recipe, which learns a compact **discrete** representation of ROP fundus images.

Why discrete latents matter here:

| Motivation | What the discrete latent enables |
| - | - |
| **Generative modelling** | A 32×32 token grid is a tractable target for a latent diffusion model or an autoregressive prior, which is far cheaper than modelling pixels directly. |
| **Data scarcity** | A learned prior over healthy and diseased retinas can synthesize additional training images for downstream stage classifiers. |
| **Representation learning** | Codebook indices are a compact, quantized descriptor that can feed classifiers, retrieval systems, or anomaly detectors. |
| **Storage and transfer** | Roughly 48× fewer elements per image than raw RGB, with the decoder acting as a learned codec. |

The repository also contains a second, independent stage: a pretrained **UNet++** that produces vessel‑structure masks for the same images, intended as an auxiliary signal or conditioning input.

> **Scope note.** This is a research prototype in a single Colab notebook, not a clinical tool. Nothing here has been validated for diagnostic use.

***

## 2. Headline results

Training ran for **46 epochs** out of a configured 200 and was stopped by an early‑stopping rule (patience 8 on validation reconstruction loss) on a Colab **T4 GPU**.

| Metric | Epoch 1 | Best | Improvement |
| - | - | - | - |
| Validation reconstruction MSE | 0.0876 | **0.0052** (epoch 38) | 94.1% lower |
| Validation FID (64‑dim features) | 11.93 | **0.1778** (epoch 39) | 98.5% lower |
| Train reconstruction MSE | 0.1567 | **0.0057** (epoch 41) | 96.4% lower |
| Train perceptual (LPIPS‑VGG) | 0.5239 | **0.0857** (epoch 41) | 83.6% lower |
| Codebook loss | 0.0776 | ~0.0890 (plateau) | stable after epoch 15 |

Two observations worth calling out:

* **Train and validation curves track each other closely for the entire run.** The gap between the two reconstruction curves stays inside noise, which is a good sign for a 2.5M‑parameter model on 2,627 images. There is no visible overfitting up to the point of early stopping.
* **The adversarial term is well behaved.** The discriminator switches on at global step 1000 (start of epoch 8) and settles into a stable band of 0.137 to 0.152, with no mode collapse and no discriminator runaway. Reconstruction loss keeps falling afterwards rather than degrading, which is the outcome you want when adding a PatchGAN to an autoencoder.

***

## 3. Repository contents

| Path | What it is |
| - | - |
| `VQVAE_ROP.ipynb` | The complete pipeline: dependency setup, data download and merge, model definitions (U‑Net blocks, VQ‑VAE, LPIPS, PatchGAN), the training loop with validation and FID, plotting, latent visualization, and the UNet++ segmentation stage. |
| `download (1).png` | 50‑row qualitative panel: original, VQ index map, reconstruction. 1018×14989 px. |
| `download (2).png` | Four‑panel training curve figure (reconstruction, perceptual, codebook, total generator loss with FID overlay). |
| `download (3).png` | 16‑tile montage of periodic training samples. Each tile shows a row of inputs above a row of reconstructions. |

The notebook is Colab‑ready and carries an **Open in Colab** badge in its first cell.

> **Housekeeping suggestion.** The three figures are still named as browser downloads. Moving them to `assets/training_curves.png`, `assets/reconstructions.png` and `assets/latent_grids.png` would make the image links in this README stable and readable. The image tags below use the current URL‑encoded names so they render as is.

***

## 4. Method

### 4.1 Pipeline at a glance

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  THREE RAW ARCHIVES (Google Drive)                                        │
 │  Stage_dataset_2.zip · dataset_model.zip · Farabi_Dataset2.zip            │
 └────────────────────────────────┬──────────────────────────────────────────┘
                                  │  merge by stage folder, convert to RGB,
                                  │  resize to 128×128, hash‑suffix filenames
                                  ▼
                    combined_dataset/  (2,627 PNG images)
                                  │
                    85 / 15 random split (seed 1111)
                                  │
        ┌─────────────────────────┴──────────────────────────┐
        ▼                                                    ▼
   train (2,232)                                        val (395)
        │                                                    │
        ▼                                                    ▼
 ┌──────────────────────────────────────────┐        recon MSE · LPIPS
 │  ENCODER    128×128×3                    │        codebook · FID
 │    conv_in        → 32ch                 │
 │    DownBlock ×1   → 64ch    stride 2     │
 │    DownBlock ×1   → 128ch   stride 2     │
 │    MidBlock  ×1   → 128ch   self‑attn    │
 │    norm + SiLU + conv_out → 3ch          │
 │    pre_quant_conv (1×1)                  │
 └────────────────────┬─────────────────────┘
                      ▼   z_e ∈ ℝ^{32×32×3}
 ┌──────────────────────────────────────────┐
 │  VECTOR QUANTIZER                        │
 │    nearest neighbour over 20 codes       │
 │    straight‑through estimator            │
 │    codebook loss + commitment loss       │
 └────────────────────┬─────────────────────┘
                      ▼   z_q, indices ∈ {0..19}^{32×32}
 ┌──────────────────────────────────────────┐        ┌────────────────────┐
 │  DECODER                                 │        │  PatchGAN          │
 │    post_quant_conv (1×1)                 │  ───▶  │  discriminator     │
 │    conv_in → 128ch                       │        │  15×15 logit map   │
 │    MidBlock ×1     self‑attn             │        │  active from       │
 │    UpBlock ×1 → 64ch   transpose conv    │        │  step 1000         │
 │    UpBlock ×1 → 32ch   transpose conv    │        └────────────────────┘
 │    norm + SiLU + conv_out → 3ch          │
 └────────────────────┬─────────────────────┘
                      ▼
              reconstruction 128×128×3
```

### 4.2 Data

Three archives are pulled with `gdown` and unzipped into a shared tree, then merged into one stage‑indexed dataset.

| Source archive | Size | Stage folders found | Notes |
| - | - | - | - |
| `Farabi_Dataset2_Zip.zip` | 554 MB | `stage0` through `stage5` | Also ships a `Plus / No Plus` split that the merge step ignores. |
| `dataset_model.zip` | 409 MB | `stage0` through `stage3` | Filenames encode clinical metadata, for example `052_M_GA26_BW820_PA31_DG1_PF0_D1_S01_1.jpg` (sex, gestational age, birth weight, postnatal age, and so on). |
| `Stage_dataset_2.zip` | 202 MB | `stage1` through `stage3` | Also contains a `laser scars` folder that the merge step ignores. |

The merge step (`Cell 5` in the notebook):

1. Iterates over `stage0` … `stage5` in each source root and skips folders that do not exist.
2. Opens each image, converts to `RGB`, resizes to **128×128** with Lanczos resampling.
3. Writes PNG output under `combined_dataset/<stage>/` with a name of the form `<original_stem>_<sha1[:10]>.png`, so identical stems from different sources never collide.

Result: **2,627 images**. The dataset class then walks every subfolder, so training is **unconditional**: stage labels organize the files on disk but are never fed to the model. Images are normalized to `[‑1, 1]` with mean 0.5 and std 0.5 per channel.

The split is `torch.utils.data.random_split` at 85/15 with global seed `1111`, giving **2,232 training** and **395 validation** images, or **140 optimizer steps per epoch** at batch size 16.

### 4.3 VQ‑VAE architecture

The encoder and decoder are built from the same U‑Net style blocks used in diffusion models (`DownBlock`, `MidBlock`, `UpBlock`), each combining GroupNorm, SiLU, 3×3 convolutions, a 1×1 residual projection, and optional multi‑head self attention. The blocks accept a time embedding argument, which is passed as `None` here and reserved for the diffusion stage.

| Component | Configuration |
| - | - |
| Input | 128×128×3, normalized to `[‑1, 1]` |
| Encoder channels | 32 → 64 → 128 |
| Downsampling | Two stride‑2 convolutions, total factor 4 |
| Encoder mid | One `MidBlock` at 128 channels with self attention |
| Attention in down blocks | Disabled (`attn_down = [False, False]`) |
| Latent channels `z_channels` | 3 |
| Latent grid | **32×32** |
| Codebook | 20 entries × 3 dimensions (60 parameters total) |
| Decoder | Mirror of the encoder, with `ConvTranspose2d` upsampling |
| Normalization | GroupNorm with 32 groups |
| Attention heads | 16 |
| Layers per block | 1 down, 1 mid, 1 up |

**Parameter counts** (measured by instantiating the model from the notebook definitions):

| Module | Parameters |
| - | - |
| Encoder including `pre_quant_conv` | 1,311,055 |
| Codebook embedding | 60 |
| Decoder including `post_quant_conv` | 1,172,431 |
| **VQ‑VAE total** | **2,483,546** |
| PatchGAN discriminator | 663,360 |

**Compression.** Each image goes from 49,152 scalar values to 1,024 discrete indices, a **48× reduction in element count**. At 5 bits per index (20 codes) against 8‑bit RGB, that is roughly **77× fewer bits**, with the decoder acting as the learned half of the codec. The very small codebook is a deliberate bottleneck: it forces the model to spend its capacity on structure rather than memorizing texture, and it keeps the token vocabulary small enough for a prior model to learn from only a few thousand images.

### 4.4 The quantizer

For each spatial position the encoder output $z_e \in \mathbb{R}^{3}$ is snapped to its nearest codebook vector under Euclidean distance, computed batch‑wise with `torch.cdist`:

$$k = \arg\min_{j \in \{1..20\}} \lVert z_e - e_j \rVert_2, \qquad z_q = e_k$$

Gradients flow through the non‑differentiable `argmin` using the **straight‑through estimator**, implemented as `z_q = z_e + (z_q - z_e).detach()`. The quantizer returns two separate losses:

$$\mathcal{L}_{\text{codebook}} = \lVert \text{sg}[z_e] - z_q \rVert_2^2 \qquad \mathcal{L}_{\text{commit}} = \lVert z_e - \text{sg}[z_q] \rVert_2^2$$

where `sg[·]` is `stop_gradient`. The first pulls the codebook toward the encoder output, the second keeps the encoder from drifting away from the codebook. It also returns the index map, which is what the latent visualization figure renders.

### 4.5 Discriminator

A standard **PatchGAN**: three strided convolution blocks at 64, 128, 256 channels with LeakyReLU(0.2) and BatchNorm, followed by a final 4×4 convolution to a single channel. For a 128×128 input it emits a **15×15** grid of logits, so the adversarial signal is local and texture oriented rather than a single global real/fake score. It is trained with **least‑squares GAN** loss (MSE against 1 and 0) rather than binary cross entropy, which is more stable at this scale.

Critically, the discriminator is **not active from the start**. It only enters the objective after `disc_start = 1000` global steps, which lands at the beginning of epoch 8. Before that the autoencoder learns pure reconstruction, so the GAN refines an already sensible image instead of fighting noise.

### 4.6 Training objective

The generator minimizes a weighted sum of four terms:

$$\mathcal{L}_{G} = \underbrace{\lVert x - \hat{x} \rVert_2^2}_{\text{recon, } w=1} + \underbrace{1.0 \cdot \mathcal{L}_{\text{codebook}}}_{\text{codebook}} + \underbrace{0.2 \cdot \mathcal{L}_{\text{commit}}}_{\text{commitment } \beta} + \underbrace{1.0 \cdot \text{LPIPS}(x, \hat{x})}_{\text{perceptual}} + \underbrace{0.5 \cdot \lVert D(\hat{x}) - 1 \rVert_2^2}_{\text{adversarial, after step 1000}}$$

The discriminator minimizes:

$$\mathcal{L}_{D} = 0.5 \cdot \tfrac{1}{2}\left( \lVert D(\hat{x}) - 0 \rVert_2^2 + \lVert D(x) - 1 \rVert_2^2 \right)$$

**Perceptual loss** is a self‑contained LPIPS implementation using a frozen ImageNet **VGG16** trunk. Features are taken from five ReLU stages (`relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`, `relu5_3`), unit‑normalized channel‑wise, squared‑differenced, passed through learned 1×1 linear probes, and spatially averaged. The v0.1 VGG linear weights are fetched from the original PerceptualSimilarity repository into `/tmp/weights/v0.1/vgg.pth`. This term is what keeps vessel edges crisp, since pixel MSE alone produces the blurry retinas that plain autoencoders are known for.

The training loop includes a graceful fallback: if the LPIPS weights fail to load, a `DummyLPIPS` module returns zeros so the run continues without the perceptual term. Check the log for the `Loading model from:` line to confirm the real one was used. In the recorded run, it was.

### 4.7 Optimization schedule

| Setting | Value |
| - | - |
| Optimizer (generator and discriminator) | Adam, β = (0.5, 0.999) |
| Learning rate | 1e‑4 for both |
| Batch size | 16 |
| Gradient accumulation | 1 step |
| Epochs configured | 200 |
| Epochs actually run | 46 (early stopped) |
| Steps per epoch | 140 |
| Total optimizer steps | ~6,440 |
| Early stopping | Patience 8 on validation reconstruction MSE |
| Checkpoints | `vqvae_autoencoder_ckpt.pth` every epoch, plus `best_vqvae_autoencoder_ckpt.pth` on each improvement |
| Sample dumps | Input/reconstruction grid every 8 steps |
| Seeds | 1111 across `torch`, `numpy`, `random`, and CUDA |
| Hardware | Single NVIDIA T4 (Colab) |

***

## 5. Results in detail

### 5.1 Training curves

<p align="center">
  <img src="download%20%282%29.png" alt="Training and validation curves for reconstruction, perceptual, codebook and total generator loss with FID overlay" width="100%">
</p>

Reading the four panels:

* **Reconstruction loss** (top left) drops steeply for the first five epochs, plateaus around 0.05 through epoch 15, then enters a second, longer descent to below 0.01. That second phase begins right after the discriminator activates. Validation tracks training throughout, with one visible spike at epoch 44.
* **Perceptual loss** (top right) falls smoothly from 0.52 to about 0.09 without a plateau, and it is the term that is still improving when training stops. Validation is noisier than training but never diverges upward.
* **Codebook loss** (bottom left) behaves the way a healthy VQ bottleneck should: a sharp rise to 0.113 by epoch 3 as codes spread out to cover the encoder distribution, then a decline and a stable plateau near 0.089 for the rest of the run. The validation trace oscillates in a narrow band around the training curve, which suggests the assignment of image regions to codes is consistent across the split.
* **Total generator loss with FID overlay** (bottom right) shows FID collapsing from 11.9 to under 1.0 by epoch 23 and continuing to about 0.18 by epoch 39. The small bump in generator loss at epochs 8 to 10 is the adversarial term entering the sum.

### 5.2 Epoch by epoch

Selected milestones from the full 46‑epoch log:

| Epoch | Train recon | Train LPIPS | Codebook | G adv | D loss | Val recon | Val FID₆₄ |
| - | - | - | - | - | - | - | - |
| 1 | 0.1567 | 0.5239 | 0.0776 | idle | idle | 0.0876 | 11.9322 |
| 3 | 0.0642 | 0.2931 | 0.1130 | idle | idle | 0.0566 | 4.3253 |
| 5 | 0.0542 | 0.2283 | 0.1052 | idle | idle | 0.0501 | 2.6405 |
| 7 | 0.0507 | 0.1965 | 0.1002 | idle | idle | 0.0463 | 2.3004 |
| **8** | 0.0510 | 0.1905 | 0.0982 | 0.0489 | 0.1622 | 0.0487 | 3.2406 |
| 12 | 0.0475 | 0.1601 | 0.0931 | 0.0380 | 0.1474 | 0.0446 | 1.7222 |
| 20 | 0.0298 | 0.1340 | 0.0864 | 0.0385 | 0.1461 | 0.0263 | 1.2059 |
| 26 | 0.0155 | 0.1203 | 0.0898 | 0.0367 | 0.1493 | 0.0116 | 0.7391 |
| 31 | 0.0093 | 0.1063 | 0.0888 | 0.0355 | 0.1511 | 0.0079 | 0.4400 |
| 37 | 0.0069 | 0.0952 | 0.0890 | 0.0358 | 0.1504 | 0.0066 | 0.2712 |
| **38** | 0.0068 | 0.0946 | 0.0895 | 0.0374 | 0.1486 | **0.0052** | 0.2361 |
| **39** | 0.0066 | 0.0930 | 0.0896 | 0.0384 | 0.1472 | 0.0062 | **0.1778** |
| 41 | **0.0057** | **0.0857** | 0.0901 | 0.0343 | 0.1520 | 0.0080 | 0.2568 |
| 46 | 0.0071 | 0.0900 | 0.0904 | 0.0441 | 0.1366 | 0.0121 | 0.2819 |

Epoch 8 is where the discriminator turns on. Epoch 38 is the best validation reconstruction and therefore the state saved to `best_vqvae_autoencoder_ckpt.pth`. Epochs 39 to 46 produce eight consecutive non‑improvements on validation reconstruction, which trips the patience counter and ends the run, even though FID and perceptual loss were still respectable in that window.

### 5.3 Reconstructions over training

<p align="center">
  <img src="download%20%283%29.png" alt="Grid of periodic training samples, inputs above reconstructions" width="100%">
</p>

Each tile is one saved checkpoint sample: the **top row is the input batch**, the **bottom row is the reconstruction**. Sample 0 and sample 1, from the first steps of training, show the expected magenta noise of an untrained decoder. Everything after that reproduces the fundus disc shape, global illumination gradient, and the characteristic colour cast of each capture device.

Note that the montage lists files in lexicographic order (`0, 1, 10, 100, 101, …`), not chronological order, so it is not a clean time series. Sorting by the integer suffix would turn this figure into a proper training progression.

### 5.4 Latent code maps

<p align="center">
  <img src="download%20%281%29.png" alt="Fifty rows of original image, VQ index map, and reconstruction" width="100%">
</p>

The tall panel shows 50 samples as **original / VQ index map / reconstruction** triplets. The middle column renders `min_encoding_indices` for the 32×32 grid through a viridis colormap with nearest neighbour upscaling, so each visible tile is exactly one token.

What to look for:

* **Codes are spatially organized, not random.** The dark border region of the fundus disc consistently maps to a distinct set of indices, and the bright optic disc region maps to another. This is the sign of a codebook that has learned meaningful structure rather than dithering.
* **The interior is high entropy.** With only 20 codes, the model spends most of its vocabulary encoding fine texture variation across the retina, producing the speckled appearance.
* **Reconstructions preserve global anatomy well and vessel detail partially.** Large vessels and the optic disc survive. The thinnest peripheral vessels are softened, and there is a faint checkerboard artifact visible near the image centre in several rows, which is the classic signature of `ConvTranspose2d` upsampling.

### 5.5 How to read the FID number

The reported FID is **not** comparable to published FID scores, and this deserves to be explicit:

1. It is computed with `FrechetInceptionDistance(feature=64)`, using the 64‑dimensional first pooling layer of Inception rather than the standard 2048‑dimensional final layer. Statistics in a 64‑dimensional space produce much smaller distances.
2. It compares **reconstructions against their own source images** on the validation set, so it measures reconstruction fidelity, not sample quality of a generative model. There is no sampling step in this pipeline yet.
3. The validation set is 395 images, well below the several thousand normally recommended for a stable FID estimate.

Read the number as a **relative training signal** that fell by two orders of magnitude, which is exactly what it is good for here, and not as an absolute claim about generation quality.

***

## 6. Vessel segmentation stage

The final section of the notebook is a separate pipeline that annotates the raw stage folders with binary masks.

| Setting | Value |
| - | - |
| Architecture | `UnetPlusPlus` from `segmentation-models-pytorch` |
| Encoder | ResNet‑18, ImageNet pretrained |
| Input | 3 channels at 256×256, ImageNet mean/std normalization |
| Output | 1 channel, sigmoid, thresholded at 0.5 |
| Weights | `best_weight_Unet++_maskresize_29` |
| Postprocessing | Mask resized back to the original resolution |
| Output files | `<original_stem>_segmented.jpg` written next to each source image |

The masks isolate branching vessel‑like structures. In the notebook's qualitative sample, coverage varies noticeably between images: one example produces a dense vessel tree, another only a few isolated fragments. That variance is expected given the wide range of illumination and focus quality in wide‑field infant fundus photography, but it means the masks should be treated as a **noisy auxiliary signal** rather than ground truth.

> **Two things to be aware of before running this section.** The checkpoint `best_weight_Unet++_maskresize_29` is **not included in this repository** and must be supplied separately at `/content/`. And because the masks are written back into the same stage folders as `_segmented.jpg`, re‑running the dataset merge afterwards would pull the masks in as training images. Run the merge first, or write masks to a separate output tree.

***

## 7. Reproducing the run

### 7.1 Fastest path

Open the notebook in Colab with a **T4 or better** GPU runtime and run the cells top to bottom. The three `gdown` cells fetch about 1.2 GB of source archives. End to end, the recorded 46‑epoch run fits comfortably in a single Colab session.

### 7.2 Environment

```bash
pip install torch torchvision
pip install lpips optuna scikit-image tqdm pyyaml pillow gdown
pip install "torchmetrics[image]" torch-fidelity
pip install segmentation-models-pytorch opencv-python matplotlib
```

Versions used in the recorded run: Python 3.12, PyTorch 2.10 with CUDA 12.8, torchmetrics 1.8.2, torchvision 0.25.

### 7.3 Execution order

| Step | Cells | What happens |
| - | - | - |
| 1 | Setup | Install and import dependencies, print device |
| 2 | Data | Three `gdown` downloads, three unzips into `/content/data/` |
| 3 | Models | Define `DownBlock`, `MidBlock`, `UpBlock`, `UpBlockUnet`, then `VQVAE` |
| 4 | LPIPS | Download VGG v0.1 weights to `/tmp/weights/v0.1/vgg.pth`, define `LPIPS` |
| 5 | Discriminator | Define PatchGAN, run the shape smoke test |
| 6 | Merge | Build `combined_dataset/`, print the image count |
| 7 | Dataset and config | Define `ROPDataset`, declare the config dictionary |
| 8 | Train | Full loop with validation, FID, checkpointing, early stopping |
| 9 | Plots | Four‑panel curve figure, sample montage, latent triplet panel |
| 10 | Segmentation | Load UNet++ weights, preview three predictions, batch write masks |

### 7.4 Gotchas

* **The LPIPS weight path is derived from where the class is defined.** The implementation resolves its weights relative to `inspect.getfile(self.__init__)`. Inside a Colab notebook that resolves under `/tmp/`, which is why the download cell targets `/tmp/weights/v0.1/vgg.pth` exactly. If you move `LPIPS` into a `.py` module, put the weights beside that module instead, or the fallback zero‑perceptual path will silently take over.
* **`ROPDataset` defaults to `im_size=28`.** The config passes 128, so the default never applies in the notebook, but it is a trap if you import the class elsewhere.
* **Images are resized twice.** The merge writes 128×128 PNGs and the dataset transform resizes to 128×128 again. Harmless, and it means you can change `im_size` without rebuilding the merged dataset.
* **The `ldm_params` block in the config is currently unused.** Nothing in the notebook trains a diffusion model. See the roadmap below.
* **The source data is not public.** The three Google Drive IDs point at private archives. To run on your own data, place images under `<root>/stage0` … `<root>/stage5` and point `src_roots` at it. Any folder of RGB images works if you adjust the stage list.

***

## 8. Full configuration reference

```python
config = {
  "dataset_params": {
    "im_path": "/content/data/combined_dataset",
    "im_channels": 3,
    "im_size": 128,
    "name": "rop"
  },
  "diffusion_params": {          # reserved for the latent diffusion stage
    "num_timesteps": 1000,
    "beta_start": 0.0015,
    "beta_end": 0.0195
  },
  "ldm_params": {                # reserved, not trained in this notebook
    "down_channels": [128, 256, 256, 256],
    "mid_channels": [256, 256],
    "down_sample": [False, False, False],
    "attn_down": [True, True, True],
    "time_emb_dim": 256,
    "norm_channels": 32,
    "num_heads": 16,
    "conv_out_channels": 128,
    "num_down_layers": 2,
    "num_mid_layers": 2,
    "num_up_layers": 2
  },
  "autoencoder_params": {
    "z_channels": 3,
    "codebook_size": 20,
    "down_channels": [32, 64, 128],
    "mid_channels": [128, 128],
    "down_sample": [True, True],
    "attn_down": [False, False],
    "norm_channels": 32,
    "num_heads": 16,
    "num_down_layers": 1,
    "num_mid_layers": 1,
    "num_up_layers": 1
  },
  "train_params": {
    "seed": 1111,
    "task_name": "rop",
    "autoencoder_batch_size": 16,
    "autoencoder_epochs": 200,
    "autoencoder_lr": 0.0001,
    "autoencoder_acc_steps": 1,
    "autoencoder_img_save_steps": 8,
    "disc_start": 1000,
    "disc_weight": 0.5,
    "codebook_weight": 1,
    "commitment_beta": 0.2,
    "perceptual_weight": 1,
    "kl_weight": 0.000005,        # used only by the VAE variant, not VQ‑VAE
    "ldm_batch_size": 64,
    "ldm_epochs": 100,
    "ldm_lr": 0.00001,
    "num_samples": 25,
    "num_grid_rows": 5,
    "save_latents": False,
    "vqvae_autoencoder_ckpt_name": "vqvae_autoencoder_ckpt.pth",
    "vqvae_discriminator_ckpt_name": "vqvae_discriminator_ckpt.pth",
    "ldm_ckpt_name": "ddpm_ckpt.pth"
  }
}
```

Artifacts land under `rop/`:

```
rop/
├── vqvae_autoencoder_ckpt.pth          # last epoch
├── best_vqvae_autoencoder_ckpt.pth     # best validation reconstruction (epoch 38)
├── vqvae_discriminator_ckpt.pth
└── vqvae_autoencoder_samples/
    └── current_autoencoder_sample_*.png
```

***

## 9. Known limitations

**Methodological**

* **No held‑out test set.** The 85/15 split gives train and validation only. Validation drives early stopping and checkpoint selection, so the reported numbers are optimistically biased. A three‑way split would fix this.
* **The split is random, not patient‑wise.** `dataset_model` filenames suggest multiple frames per infant. A random split can place frames from the same eye in both partitions, which inflates apparent reconstruction quality. **Grouping by patient ID before splitting is the single highest‑value change to make.**
* **No codebook utilization tracking.** With 20 entries, dead codes are a real risk and would be invisible in the current logs. A per‑epoch histogram of index usage and a perplexity metric would confirm whether all 20 are alive.
* **No SSIM or PSNR.** Both are imported at the top of the notebook and never used, even though they are the standard companions to MSE for image reconstruction and would make the results directly comparable to other work.
* **FID caveats.** See section 5.5.
* **Optuna is imported but no search is run.** Codebook size, commitment β, perceptual weight, and `disc_start` are all obvious candidates for a study.

**Engineering**

* Everything lives in one notebook, with model definitions, training, and evaluation interleaved. There is no importable module, no CLI, no test.
* Extra optimizer steps run at the end of each epoch after the inner loop has already stepped, which applies whatever gradients happen to be sitting in the buffers.
* The latent visualization cell builds a **fresh** model and loads the checkpoint with `strict=False`, so a silent partial load would not raise an error. Since the checkpoint is RGB and the fresh model is RGB, the load is complete in practice, but `strict=True` would make that guarantee explicit.
* The UNet++ checkpoint is not versioned in the repository, so the segmentation section is not reproducible as published.
* Stage labels are discarded. They are already on disk and would cost nothing to carry through for conditional modelling.

***

## 10. Roadmap

The configuration file makes the intended direction clear. In rough priority order:

1. **Patient‑wise splitting and a held‑out test set.** Prerequisite for any number in a paper.
2. **Codebook diagnostics.** Log index histograms and perplexity. If codes are dead, add EMA codebook updates or code restarts, both standard fixes.
3. **Report SSIM and PSNR** alongside MSE and LPIPS, and add a standard FID at `feature=2048` for comparability.
4. **Train the latent diffusion stage.** `ldm_params` and `diffusion_params` already describe a 128 to 256 channel U‑Net with attention at every resolution and a 1000‑step linear beta schedule from 0.0015 to 0.0195, operating on the 32×32 latent grid. The `UpBlockUnet` class with cross attention support is already defined and unused, which is exactly what conditioning on stage labels or vessel masks would need.
5. **Condition on stage.** The folder structure already encodes stages 0 to 5. Class‑conditional generation would let you synthesize targeted examples for the underrepresented severe stages.
6. **Condition on vessel masks.** The UNet++ stage exists precisely because vessel structure is the clinical signal in ROP. Feeding masks through the cross attention path is the natural next experiment.
7. **Downstream validation.** The honest test of a medical generative model is whether synthetic data improves a real classifier. Train a stage classifier on real data, then on real plus synthetic, and report the delta.
8. **Ablations.** Codebook size, the effect of removing the perceptual term, and the effect of removing the discriminator, each measured on the same held‑out set.

***

## 11. Suggested layout for a refactor

```
VQVAE_ROP/
├── README.md
├── requirements.txt
├── configs/
│   └── rop_vqvae.yaml
├── src/
│   ├── models/
│   │   ├── blocks.py            # DownBlock, MidBlock, UpBlock, UpBlockUnet
│   │   ├── vqvae.py             # VQVAE
│   │   ├── discriminator.py     # PatchGAN
│   │   └── lpips.py             # perceptual loss
│   ├── data/
│   │   ├── prepare.py           # merge and resize into combined_dataset
│   │   └── dataset.py           # ROPDataset
│   ├── train_vqvae.py
│   ├── train_ldm.py             # the next stage
│   └── evaluate.py              # MSE, SSIM, PSNR, LPIPS, FID, codebook stats
├── scripts/
│   └── segment_vessels.py       # UNet++ mask generation
├── assets/
│   ├── training_curves.png
│   ├── reconstructions.png
│   └── latent_grids.png
└── notebooks/
    └── VQVAE_ROP.ipynb          # kept as the exploratory record
```

***

## 12. References

| Work | Relevance |
| - | - |
| van den Oord, Vinyals, Kavukcuoglu (2017), *Neural Discrete Representation Learning* | The original VQ‑VAE, the quantizer and straight‑through estimator used here |
| Esser, Rombach, Ommer (2021), *Taming Transformers for High‑Resolution Image Synthesis* | VQGAN: the perceptual plus adversarial training recipe this project follows |
| Rombach et al. (2022), *High‑Resolution Image Synthesis with Latent Diffusion Models* | The target of the roadmap: diffusion in the learned latent space |
| Zhang et al. (2018), *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* | LPIPS, including the v0.1 VGG weights loaded by the notebook |
| Isola et al. (2017), *Image‑to‑Image Translation with Conditional Adversarial Networks* | PatchGAN discriminator |
| Zhou et al. (2018), *UNet++: A Nested U‑Net Architecture for Medical Image Segmentation* | The vessel segmentation stage |
| Heusel et al. (2017), *GANs Trained by a Two Time‑Scale Update Rule Converge to a Local Nash Equilibrium* | FID |

**Libraries**: PyTorch, torchvision, torchmetrics, `segmentation-models-pytorch`, `lpips`, `torch-fidelity`, scikit‑image, Optuna.

***

## Acknowledgements

Fundus imagery originates from ROP screening datasets including material referenced as the Farabi dataset. Clinical data of this kind is collected under institutional approval and is not redistributed here.

## Citation

```bibtex
@software{poor_vqvae_rop,
  author = {Amirhossein Poor},
  title  = {VQVAE_ROP: Discrete latent representation learning for
            Retinopathy of Prematurity fundus imaging},
  year   = {2026},
  url    = {https://github.com/Amirhosseinpoor/VQVAE_ROP}
}
```

## License

No license file is currently present, which means default copyright applies and others have no permission to reuse the code. Adding an `MIT` or `Apache‑2.0` license file would make the intent explicit.

***

<p align="center">
  <sub>Research prototype. Not a medical device and not validated for diagnostic use.</sub>
</p>
