
  This repository accompanies our paper: **“DAWN: Spatiotemporal Self-Supervision Enables Long-Term, Low-Photon Single-Molecule Fluorescence Measurements.”** 
  DAWN provides a universal self-supervised computational framework for high-fidelity imaging at the photon limit, enabling the study of molecular dynamics with unprecedented duration and minimal photodamage.

The repository is organized into three main components for easy use and custom development:

* **DAWN Train GUI code**
  The source code for the DAWN training GUI.
  A user-friendly interface for training DAWN models. With just a few clicks, you can complete the entire training workflow, save checkpoints, and monitor loss, GPU usage, learning rate, and more.

* **DAWN Inference GUI code**
  The source code for the DAWN inference GUI.
  Designed for applying trained DAWN checkpoints to test data. It also allows real-time monitoring of CPU/GPU usage during inference.

* **DAWN Source Code**
  Contains the full codebase for DAWN training/testing. This component is fully customizable to support your own extensions or modifications.

  Detailed instructions and important precautions for each part are provided in the corresponding `**User_Guide.md` files.

  For convenience, we also provide ready-to-use GUI versions that do not require additional setup. You can download them from **Tsinghua Cloud**:

* [Train GUI (Tsinghua Cloud)](https://cloud.tsinghua.edu.cn/d/8abf43bdada243d69822/)
* [Inference GUI (Tsinghua Cloud)](https://cloud.tsinghua.edu.cn/d/f3ae5bd279b140aa8640/)

  We would be honored if you gave us a star ⭐ if you find our method useful!


