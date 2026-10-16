# Release scope and provenance

This repository contains a source-only REACT2025 release: the FlowREACT V41 model, 3DMM-to-emotion converter, Hydra configurations, training and evaluation code, analysis utilities, and focused tests. Generated outputs, REACT2024 adapters, and review documents are excluded.

The included `LICENSE` is the notice shipped with the REACT2025 baseline code. FaceVerse, PIRender, challenge datasets, and their trained assets have separate terms and are not bundled. Obtain those resources from their respective projects before using optional rendering or running the data loaders. Do not add dataset archives, face videos, checkpoints, or other restricted assets when publishing this folder.

The local workspace also contains a temporal-GAN **fine-tuning** launcher that initializes from an existing FlowREACT checkpoint. It does not implement a from-scratch, GAN-only replacement of the original objective, so it is not presented in this release as code for that reported comparison. Add the verified GAN-only training implementation separately if that experiment is to be reproduced from this repository.

The source configurations default to an eight-frame REACT2025 window. A checkpoint and its evaluation must use the same recorded settings; this packaging step does not alter training behavior or experimental results.

The README commands and source/configuration links have been checked statically. A full CUDA training run and metric evaluation require the separately supplied datasets, model files, and Python environment.
