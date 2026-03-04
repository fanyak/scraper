A scraper for wikipedia pages using Simon Willison's shot-scraper package.
Can be run with streamlit

For Streamlit Cloud deployment:
- Keep Python dependencies in `requirements.txt` (includes `playwright`).
- Keep OS dependencies in `packages.txt` (required by Playwright Chromium).

The app installs Chromium at runtime using:
`python -m playwright install chromium`
