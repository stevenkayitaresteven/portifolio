"""Fine-tuning suite for a custom whole-image explicit-content classifier.

This package lets you train your *own* classifier on labeled data and plug it
into the runtime via ``ClassifierDetector`` (set ``use_classifier=True`` and
``classifier_path``). It is intentionally separate from the inference core so the
runtime never needs PyTorch unless you opt into the learned model.

See ``safety/train/README.md`` for the full workflow. The short version::

    # 1. organize data as  data/{train,val}/{safe,nudity,gore}/*.jpg
    python -m safety.train.train --data ./data --epochs 8 --out runs/v1
    # 2. export to ONNX so inference needs only onnxruntime (no torch)
    python -m safety.train.export runs/v1/best.pt --out runs/v1/model.onnx
    # 3. use it
    python -m safety blur ./photos --classifier runs/v1/model.onnx
"""
