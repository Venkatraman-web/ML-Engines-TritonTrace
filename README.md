# DARTIS 2019 Oil Slick vs Look-alike Classifier

A small PyTorch classifier that separates **oil slicks** from **look-alikes** (natural films, low-wind areas, etc.) using the spatial metadata in the **DARTIS 2019** dataset (Sentinel-1, Eastern Mediterranean). It is part of the TritonTrace oil-spill forensic platform.

![DARTIS Oil Spill Detector UI](ui_preview.jpg)

## How it works

| Step | Details |
|---|---|
| **Data** | `DARTIS_2019 (1).tab` (place it next to the script) |
| **Features** | Patch width and height, object bounding box (x min/max, y min/max) and label size, all in pixels |
| **Target** | `1` = oil (`ow`, `oc` subsets), `0` = non-oil (`nw`, `nc` subsets) |
| **Preprocessing** | 80/20 stratified split, z-score normalisation using training-set statistics |
| **Model** | MLP `7 → 64 → 32 → 1` (ReLU), `BCEWithLogitsLoss`, Adam (lr 0.01), 150 epochs |
| **Prediction** | `predict_raw_inputs(...)` returns the label, confidence and raw probability |

## Usage

Written for Jupyter / Google Colab.

```bash
pip install torch numpy pandas scikit-learn
```

1. Put `DARTIS_2019 (1).tab` in the same folder as `dartis_classifier.py`.
2. Run the script in a notebook cell. It trains the model, then renders the HTML form shown above.
3. To classify from Python:

```python
predict_raw_inputs(640, 640, 345, 297, 368, 343, 1058)
```

## Files

- `dartis_classifier.py` — data loading, training, prediction helper and notebook demo UI
- `ui_preview.jpg` — screenshot of the demo form

## Limitations

- The model learns from **dataset metadata**, not SAR image pixels.
- The "Classify Metadata" button in the HTML form uses a simple rule in JavaScript for demo purposes; the trained model is the Python `predict_raw_inputs` function.
- No evaluation metrics are computed or reported in the script.
