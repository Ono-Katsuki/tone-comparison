# VOICE Model Comparison Demo

A side-by-side comparison tool for three predictive-text models — two
personalized models featured in the paper's demo video (an on-device
Gemma 3 270M SFT+DPO model and a Gemini 2.5 Flash SFT model, v11) plus
an untuned Gemini 2.5 Flash baseline. The paper reports on additional
tuned versions that are not run in this app. The app is used to record
the 37-second demo video accompanying:

> Katsuki Ono. 2026. **Reconfiguration of Ability: An Autoethnography of
> LLM-Mediated Development by a Developer with Severe Physical Disability.**
> In *The 28th International ACM SIGACCESS Conference on Computers and
> Accessibility (ASSETS '26).* ACM. https://doi.org/10.1145/3797867.3828990

See `CITATION.cff` for a machine-readable citation.

> **Note on model lifecycle.** The `gemini-2.5-flash` identifier used
> below (both for the tuned endpoint and for the base baseline) is
> scheduled for shutdown on **October 16, 2026** (paper §5.8; Gemini
> API deprecations page). Treat this demo as a reference for the
> comparison logic — swap in a currently-supported Gemini release
> when running it yourself.

## Models compared

| Model | Backing | Description |
|-------|---------|-------------|
| **Local** | On-device | [`katsukiono/gemma3-270m-pred-dpo`](https://huggingface.co/katsukiono/gemma3-270m-pred-dpo) — 270M-parameter Gemma 3 model fine-tuned for predictive text (`pred`) with SFT and DPO. |
| **Tuned (v11)** | Vertex AI endpoint | Eight tones (`dev` / `meeting` / `casual` / `business` / `polite` / `friendly` / `concise` / `enthusiastic`) generated in parallel; for this side-by-side demo, the three most diverse candidates are returned (the deployed keyboard uses a different selection — one candidate per tone; see paper §4.8). |
| **Gemini Flash** | Vertex AI | `gemini-2.5-flash` — the base model that Tuned (v11) was fine-tuned from; shown as a no-fine-tuning baseline. |

## Setup

### Requirements

```bash
pip install flask google-genai torch transformers accelerate
```

(`accelerate` is required because the Local Gemma model is loaded with
`device_map='auto'`.)

### Vertex AI authentication

```bash
gcloud auth application-default login
```

### Configuration (environment variables)

The Vertex AI project ID, region, and tuned-endpoint path are read from
environment variables. The code ships with **safe placeholder defaults**;
set all three variables for your own Vertex AI deployment.

```bash
export VERTEX_PROJECT="<your-gcp-project-id>"
export VERTEX_LOCATION="us-central1"
export TUNED_ENDPOINT="projects/<PROJECT_NUMBER>/locations/us-central1/endpoints/<ENDPOINT_ID>"
```

The author's own `TUNED_ENDPOINT` is a personalized Vertex AI endpoint
that a third party cannot access — reproducing "Tuned (v11)" against
your own data requires first running the SFT pipeline in
[`Project-VOICE-SFT`](https://github.com/Ono-Katsuki/Project-VOICE-SFT)
against your own corpus, then pointing `TUNED_ENDPOINT` at the resulting
endpoint.

### Run

```bash
python app.py
```

Open the demo at http://127.0.0.1:5001/.

## Structure

```
tone-comparison/
├── app.py                      # Flask server (port 5001)
├── index.html                  # Frontend (kana keyboard + suggestion bars)
└── static/
    └── tiny_segmenter-0.2.js   # Japanese word segmenter (BSD, © Taku Kudo)
```

## Usage

1. Type hiragana on the 50-sound Japanese kana keyboard.
2. All three models' predictions are shown in real time.
3. Each model reports its own response time in milliseconds; the fastest
   response gets a green badge.
4. Clicking a prediction inserts it into the composition.

## Companion repository

The SFT pipeline that produced the **Tuned (v11)** Gemini 2.5 Flash
endpoint is at
[Ono-Katsuki/Project-VOICE-SFT](https://github.com/Ono-Katsuki/Project-VOICE-SFT).
The **Local** Gemma 3 270M SFT+DPO weights come from a separate
training notebook shipped with the paper's supplementary bundle.

## Third-party notices

This project bundles TinySegmenter (`static/tiny_segmenter-0.2.js`,
BSD 2-Clause, © 2008 Taku Kudo). See `NOTICE`.

## Cited BibTeX

```bibtex
@inproceedings{ono2026reconfiguration,
  author    = {Katsuki Ono},
  title     = {Reconfiguration of Ability: An Autoethnography of {LLM}-Mediated
               Development by a Developer with Severe Physical Disability},
  booktitle = {The 28th International ACM SIGACCESS Conference on Computers
               and Accessibility (ASSETS '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3797867.3828990},
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
