# 📄 AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get an estimated **ATS score**, a category breakdown, missing keywords, formatting problems, prioritised improvements and rewritten example bullets. Optionally paste a job description for a role-specific score.

Built with **Streamlit** and **Google Gemini Flash**.

## Features

- Reads PDF, DOCX (including tables) and TXT resumes
- Overall ATS score (0-100) calculated from five weighted categories:
  Keywords & Relevance (30%), Formatting & Structure (20%), Content Quality (20%), Impact & Metrics (20%), Readability (10%)
- Found vs. missing keywords
- High / Medium / Low priority improvements with concrete fixes
- Before/after bullet-point rewrites
- "What an ATS sees" view showing the extracted text
- Downloadable Markdown report
- Optional job-description matching

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. **Install Python 3.10+** and clone this repository.
2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Get a free Gemini API key** at <https://aistudio.google.com/app/apikey>.
5. **Provide the key** in one of these ways:
   - Paste it in the app's sidebar (simplest), or
   - Create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     # GEMINI_MODEL = "gemini-2.5-flash"   # optional
     ```
   - Or set an environment variable: `export GEMINI_API_KEY="your-key-here"`
6. **Start the app**
   ```bash
   streamlit run app.py
   ```

## Configuration

| Setting | Where | Description |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | Your Gemini API key |
| `GEMINI_MODEL` | secrets / env var (optional) | Default model. Falls back to `gemini-2.5-flash` |

### Choosing a model

The sidebar model list is **loaded live from your API key**, so it only shows Flash models you can actually use. If the live list can't be loaded, a short list of common models is shown instead. You can also type any model ID under **Custom...**.

If the selected model returns a 404 (model not found), the app automatically retries with other available Flash models and shows which one answered. Model IDs must match exactly: for example `gemini-3.0-flash` does not exist, while `gemini-3-flash-preview` and `gemini-2.5-flash` do. See the [Gemini models list](https://ai.google.dev/gemini-api/docs/models) for current IDs.

## Deploy on Streamlit Community Cloud

1. Push this project to a **GitHub repository**.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main` and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

> ⚠️ If you store your key in Secrets on a **public** app, anyone using it spends **your** quota. Make the app private (Share settings) or leave the secret empty so each visitor enters their own key in the sidebar.

## Security & privacy

- **Never commit your API key.** Add this to a `.gitignore` file:
  ```
  .streamlit/secrets.toml
  venv/
  __pycache__/
  .env
  ```
- Resumes are sent to Google's Gemini API for analysis. The app itself does not save uploaded files.
- Resume and job-description text is treated as data, so instructions hidden inside a resume cannot change the scoring, and the overall score is calculated in code from the category scores.

## Limitations

- The score is an **AI estimate**. Every real ATS (Workday, Greenhouse, Lever, Taleo…) works differently, so treat it as guidance, not a guarantee.
- Scanned or image-only PDFs have no extractable text; export a text-based PDF or DOCX instead.
- Very long resumes are truncated to the first ~30,000 characters.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Analysis failed: ... API key not valid` | Re-check the key; no spaces or quotes in the sidebar box |
| `Analysis failed: ... 404 NOT_FOUND` / "No usable model was found" | The model ID doesn't exist for your key. Pick one from the sidebar list (loaded from your key) |
| `429 / quota exceeded` | Wait a minute or upgrade your Gemini plan |
| "Very little text could be extracted" | Your PDF is probably an image; export it as a text PDF or DOCX |
| `ModuleNotFoundError` on Streamlit Cloud | Make sure `requirements.txt` is in the repo root |

## License

MIT - use it freely.
