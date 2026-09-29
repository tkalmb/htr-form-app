# htr-form-app

A Streamlit app that extracts handwritten entries from scanned German administrative forms. It accompanies the master's thesis *Handwritten Text Recognition for German Administrative Forms* and runs the same pipeline code (`htrpipe`) that the thesis evaluation used.

The app guides the user through six steps: upload scans or a PDF, define the form fields, choose a model (fine-tuned TrOCR, Qwen3-VL-8B or Chandra OCR 2) and preprocessing preset, complete the model-specific setup, run the extraction, then review, correct and export the results. Implausible or low-confidence values are flagged for review, never changed automatically.

It was developed and tested only on the three layouts of the HGAF dataset ([Zenodo](https://doi.org/10.5281/zenodo.22934701)). Other forms can be defined in the app, but its behaviour on them has not been evaluated. For other forms, the lexicons and word lists in `resources/` may need to be replaced. They are plain `.csv`/`.txt` files and can be swapped without code changes.

## Folder structure

```
app.py              Streamlit UI
appcore/            glue between the UI and the pipeline (inputs, presets, model engines, flags, manifest)
htrpipe/            the thesis pipeline package, trimmed to the modules the app uses
layouts/            form configurations for HGAF layouts A, B and C
resources/          lexicons and word lists used by post-processing
config.yaml         model paths and output directories
requirements.txt    pinned dependencies
```

## How to run

Requires Python 3.12 and a CUDA GPU with at least 24 GB VRAM for the vision-language models.

```bash
conda create -n htr-form-app python=3.12
conda activate htr-form-app
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
streamlit run app.py
```

All model weights are downloaded from Hugging Face on first use: the fine-tuned TrOCR checkpoint ([`tkalmb/trocr-HGAF`](https://huggingface.co/tkalmb/trocr-HGAF)), Qwen3-VL-8B and Chandra OCR 2. To use a local copy instead, point the corresponding path in `config.yaml` to it.

On a remote server, forward the port and open `http://localhost:8501` locally:

```bash
ssh -L 8501:localhost:8501 user@server
```

## Licence

MIT, see `LICENSE`. The `german_cities` lexicon derives from Wikipedia's [List of cities and towns in Germany](https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Germany) (CC BY-SA 4.0).
