# Weights licence: non-commercial use only

This covers `parts.safetensors` and `flight.safetensors` from https://github.com/lincleejun/bioclip-bird-heads.
The code in that repository is MIT (LICENSE); the weights are not. Whoever uses the weights takes on the terms of
the data they were trained on, which do not allow commercial use:

- `parts.safetensors` is trained on **CUB-200-2011** (Caltech-UCSD Birds), whose images are provided for
  non-commercial research and educational purposes only.
- `flight.safetensors` is trained on iNaturalist photos: 4,245 of the 4,883 listed are **CC BY-NC**, 528 CC BY,
  110 CC0. Each photo's licence and attribution is in the download manifest `fetch_flight.py` writes.
- Both are built on BioCLIP 2.5 (`imageomics/bioclip-2.5-vith14`, MIT) and need its image tower to run.
- These terms are our reading of the data's licences, not legal advice; check them before any use beyond
  personal research.
