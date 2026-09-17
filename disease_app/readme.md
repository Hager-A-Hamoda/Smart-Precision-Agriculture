# Leaf Disease Classifier (Streamlit)

A minimal Streamlit app that deploys **only** the two disease-classification
models from the Smart-Precision-Agriculture project:

| Fruit  | Model file                          | Classes |
|--------|--------------------------------------|---------|
| Apple  | `best_model FF.keras`                | Healthy, Apple Scab, Black Rot, Cedar Apple Rust |
| Tomato | `mobilenetv2_stage1_best.keras`      | Healthy, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot |

(Class order and preprocessing were taken directly from the project's
original `app.py` so predictions line up with how the models were trained/used.)

## 1. Folder layout

`app.py` is set up for your current layout, where the app lives
in a `disease_app/` subfolder one level below the repo root:

```
Smart-Precision-Agriculture/          <- repo root
├── disease_app/
│   ├── app.py              <- this app
│   ├── DiseaseAppRequirements.txt
│   └── readme.md
├── best_model FF.keras               <- already in repo (apple disease model)
└── tomatoDisease/
    └── mobilenetv2_stage1_best.keras <- already in repo (tomato disease model)
```

The paths are simple relative strings resolved against the repo root
(`disease_app/`'s parent folder):

```python
APPLE_DISEASE_PATH = "best_model FF.keras"
TOMATO_DISEASE_PATH = r"tomatoDisease\mobilenetv2_stage1_best.keras"
```

If for some reason a model isn't found at that exact spot, the app falls
back to searching the rest of the repo for a file with the same name.

## 2. Install dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r disease_app\DiseaseAppRequirements.txt
```

### Python 3.13 note

`requirements.txt` pins `tensorflow>=2.21`, which is the first TensorFlow
release with official Python 3.13 wheels (3.9 was dropped in the same
release; 3.10–3.13 are supported). Older pins like `tensorflow<2.17` will
fail to install on Python 3.13 — there's simply no matching wheel — so if
`pip install` can't find a TensorFlow version for you, upgrade pip first
(`pip install --upgrade pip`) and make sure nothing else in your venv is
forcing an older TensorFlow.

You do **not** need to switch to Python 3.11 for this app.

> Note: if you still get an error loading either `.keras` file (e.g. an
> unknown layer/config error), it usually means the model was saved with a
> very different Keras version than the one that ships with your installed
> TensorFlow. The `.keras` format (Keras 3's native format) is generally
> forward-compatible, but if you hit issues, check what TF/Keras version the
> models were trained with and match it as closely as possible.

## 3. Run the app

```bash
cd disease_app
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## 4. Using it

1. Choose the **🍎 Apple** or **🍅 Tomato** tab.
2. Upload a leaf image (JPG/PNG).
3. Click **Run Prediction**.
4. You'll see the predicted class, confidence, and a bar chart of all
   class probabilities.

## 5. Deploying (optional)

- **Streamlit Community Cloud**: push this folder (with the models — use
  Git LFS if the `.keras` files are large) to a GitHub repo and deploy from
  streamlit.io/cloud.
- **Hugging Face Spaces**: create a Space with SDK "Streamlit", upload these
  files plus the `models/` folder.

## Notes / troubleshooting

- If a model file isn't found, the app raises a clear error telling you the
  expected filename — double check it's inside `models/`.
- Both models are loaded lazily and cached (`st.cache_resource`), so they're
  only loaded once per session, not on every prediction.
- Preprocessing: images are resized to match each model's own input shape
  (read from the model itself) and scaled to `[0, 1]`. This mirrors the
  original project's `preprocess_for_shape` function.
- If a model turns out to have a single sigmoid output instead of a softmax
  over all classes, the app treats it as binary (class 0 vs class 1) using
  the first two entries of the class list — you may need to adjust
  `APPLE_DISEASE_CLASSES` / `TOMATO_DISEASE_CLASSES` order if that's the case
  for your model file.