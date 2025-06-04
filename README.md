# REGAL: Rules-based Explanations for Generated neighborhoods Around Localized cases

This repository contains the core implementation of **REGAL**, a novel explainable AI framework for interpreting multimodal deep learning models. REGAL is designed to generate regionally faithful explanations by integrating realistic synthetic sampling (via CT-GAN), feature-level attribution, and image-level saliency using Grad-CAM. It was developed as part of my culminating master's thesis project to support explainability in search and rescue (SAR) outcome prediction.

> ⚠️ **Note:** Due to data privacy agreements, the dataset used in this work cannot be shared. This repository is intended to document code structure and logic, not to support direct execution.

---

## 📁 Repository Structure


---

## Key Components

### `data_processing/`
Scripts responsible for:
- Cleaning and preprocessing the original MRA dataset (obtained from Isrid2, provided by Robert Koester) 
- Scraping historical weather data using a OpenMeteo API 
- Feature engineering geographical context (e.g. state, county, city)

### `succsarnet.py`
Implements **SuccSARNet**, a multimodal neural network that fuses:
- ResNet-based embeddings from satellite images
- A fully connected network for tabular features
- A fusion MLP for binary classification (success vs. failure in SAR)

This script handles model training, evaluation, and internal embedding extraction.

### `REGAL.py`
Contains all core explainability methods, including:
- **CT-GAN neighborhood generation**: Realistically simulates a generated neighborhood near an instance
- **Feature attribution**: Uses a decision tree explainer on synthetic neighborhoods to assess importance and interaction
- **Image saliency**: Applies Grad-CAM to highlight spatial regions of important embeddings relevant to predictions

All functions are modular and designed for interpretability pipelines. This file does not include a CLI or runnable script due to data limitations.

---

## Limitations

- **No public dataset**: All data is protected under an NDA and cannot be shared.
- **Not immediately executable**: This repository is intended as a code showcase, not a plug-and-play tool.
- **Environment setup and training scripts** are not provided, as replicability requires access to private data.

---

## 📜 Citation

If you use or build upon this code, please cite the corresponding thesis:

> Kim, B. (2025). *Opening the Black Box with REGAL: A Novel Explainable AI Approach to Uncover Key Predictors in Search and Rescue Success*. California Polytechnic State University, San Luis Obispo.

---

## Acknowledgements

Special thanks to the Cal Poly Statistics Department and CSSE Department and all contributors to this research effort. Satellite image processing was enabled by pretrained ResNet architectures; weather data was obtained via Open-Meteo’s public API.

---

## Contact

For academic inquiries or collaborations, feel free to reach out via email or [LinkedIn](https://www.linkedin.com/in/brandon-kim1/).

