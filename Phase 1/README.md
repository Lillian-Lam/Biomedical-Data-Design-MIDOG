# Biomedical Data Design II: MIDOG Phase 1
## Project Overview

This repository contains the code and documentation for **Phase 1** of our Biomedical Data Design II project, focused on **domain shift quantification, feature extraction, and similarity analysis** for the MIDOG histopathology dataset.

Mitotic figure detection in histopathology images is a challenging computer vision task because model performance can vary significantly across different domains, including scanner type, laboratory source, animal species, and tumor type. Before building or fine-tuning a detection model, this phase aims to quantify how different image domains relate to one another and identify which feature representations best capture meaningful inter-domain variation.

The goal of this phase is to provide a data-driven foundation for later model development by measuring domain differences, comparing feature extractors, and visualizing relationships between subsets of the dataset.

## Repository Structure

```text
Phase 1/
│
├── Domain Shift Quantification/
│   └── Code and documentation for measuring domain shift using statistical and distribution-based methods
│
├── Feature_extractors/
│   └── Scripts for extracting image embeddings using pretrained models
│
├── Similarity/
    └── Code for generating similarity matrices and comparing domains

