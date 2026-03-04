import html
import json
import re
import string
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from nltk.corpus import stopwords
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from wordcloud import WordCloud


DEFAULT_SCRIPT = """(() => { function tableToObjects() {const nodes = Array.from(document.querySelectorAll("section:first-child table, section:first-child p, .mw-heading ~ p, .mw-heading ~ ul, .mw-heading ~ table, ol.references, h1, h2, h3, h4")); const obj = {}; let content = []; let title = ""; const order = []; const tags = []; for (let i = 0; i < nodes.length; i++) { const node = nodes[i]; if (node.id && ["H1", "H2", "H3", "H4"].includes(node.tagName)) {content = [];  title = node.innerText; order.push(title); tags.push(node.tagName?.toLowerCase())} else {content.push(`<${node.tagName.toLowerCase()}>${node.getHTML()}</${node.tagName.toLowerCase()}>`)};  obj[title] = content } return ({obj, order, tags});} return tableToObjects();})"""

with open("language_map.json", "r", encoding="utf-8") as f:
    LANGUAGE_MAP = json.load(f)


@st.cache_resource
def ensure_nltk_stopwords() -> None:
    nltk.download("stopwords", quiet=True)


def ensure_script_file(script_path: Path) -> None:
    if not script_path.exists():
        script_path.write_text(DEFAULT_SCRIPT, encoding="utf-8")


@st.cache_resource
def ensure_playwright_chromium() -> None:
    # sys.executable is the path to /bin/python that the app runs with, 
    # which ensures the playwright installation is in the same environment
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Failed to install Playwright Chromium: {details}")


def run_playwright_fallback(url: str, script_path: Path) -> dict:
    script = script_path.read_text(encoding="utf-8")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            data = page.evaluate(script)
        finally:
            context.close()
            browser.close()
    if not isinstance(data, dict):
        raise ValueError("Fallback scraper did not return structured JSON data.")
    return data


def run_shot_scraper(url: str, script_path: Path) -> dict:
    ensure_script_file(script_path)
    ensure_playwright_chromium()
    try:
        result = subprocess.run(
            ["shot-scraper", "javascript", "-i", str(script_path), url],
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            lines = [line for line in output.splitlines() if line.strip()]
            if not lines:
                raise ValueError("Shot-scraper did not return any JSON output.")
            return json.loads(lines[-1])
    except subprocess.CalledProcessError as exc:
        stderr_text = (exc.stderr or exc.stdout or "").strip()
        if "Page.goto: Page crashed" not in stderr_text:
            raise
        return run_playwright_fallback(url, script_path)
    except PlaywrightError:
        return run_playwright_fallback(url, script_path)


def parse_content(data: dict) -> dict[str, str]:
    parsed_content: dict[str, str] = {}
    for node in data["order"]:
        parsed_content[node] = ""
        for paragraph in data["obj"].get(node, []):
            parsed_content[node] +=  re.sub(r"\{\{\{.*?\}\}\}", "", paragraph, flags=re.S)
    return parsed_content


def build_html_page(data: dict, parsed_content: dict[str, str]) -> str:
    page = ""
    for idx, name in enumerate(data["order"]):
        tag = data["tags"][idx].lower()
        segment = f"<{tag}>{name.replace('_', ' ')}</{tag}>\n{parsed_content[name]}"
        page += segment
    return f"""<!doctype html>
        <html>
            <head>
                <meta charset=\"utf-8\" />
                <style>
                    body {{
                        background-color: #ffffff;
                        margin: 0;
                        padding: 12px;
                    }}
                </style>
            </head>
            <body>
                {page}
            </body>
        </html>"""


def unify_content(parsed_content: dict[str, str]) -> str:
    return "".join(parsed_content.values())


def clean_text_content(text: str) -> str:
    text = re.sub(r"<figure>.*?</figure>", "", text, flags=re.S)
    text = re.sub(r"<img>.*?</img>", "", text, flags=re.S)
    text = re.sub(r'<style .*?</style>', '', text, flags=re.S)
    text = re.sub(r"\{\{\{.*?\}\}\}", "", text, flags=re.S)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_word_counts(text: str) -> Counter:
    return Counter(text.split())


def get_stopwords_for_content(url: str, fallback_language: str = "english") -> set[str]:
    lang_prefix = url.split("//")[-1].split(".")[0]
    language = LANGUAGE_MAP.get(lang_prefix, fallback_language)
    try:
        return set(stopwords.words(language))
    except LookupError:
        return set(stopwords.words(fallback_language))


def filter_by_stopwords(word_counts: Counter, stopword_set: set[str]) -> Counter:
    filtered = Counter()
    for word, count in word_counts.items():
        if word not in stopword_set and len(word) > 3:
            filtered[word] = count
    return filtered


def main() -> None:
    st.set_page_config(page_title="Wikipedia Shot Scraper Analyzer", layout="wide")
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"] {
                max-height: none !important;
                overflow-y: visible !important;
            }

            div[role="tabpanel"] {
                max-height: calc(100vh - 100px);
                overflow-y: auto !important;
                padding-bottom: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Wikipedia Shot Scraper Analyzer")
    st.write("Scrape a Wikipedia page and analyze its content.")

    ensure_nltk_stopwords()
    script_path = Path("get_pdf.js")

    # st.header("User Input (wikipedia page url)")
    default_url = "https://als.wikipedia.org/wiki/Schweiz"
    url = ""

    if "scraped_url" not in st.session_state:
        st.session_state.scraped_url = ""
    if "parsed_content" not in st.session_state:
        st.session_state.parsed_content = None
    if "html_page" not in st.session_state:
        st.session_state.html_page = ""
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    
    scrape_tab, analyze_tab = st.tabs(["Scrape Page Content", "Analyze content"])

    # with user_input_tab:
    #     st.header("User Input (wikipedia page url)")
    #     url = st.text_input("Wikipedia page URL", value=default_url)

    with scrape_tab:
        st.header("Scrape Page Content")
        url = st.text_input("Enter a Wikipedia page URL", value=default_url)

        scrape_clicked = st.button("Run scrape")

        if scrape_clicked:
            if not url.strip():
                st.error("Please enter a valid Wikipedia URL.")
            elif re.search(r"https://\w+\.wikipedia\.org/wiki/.+", url.strip()) is None:
                st.error("Please enter a valid Wikipedia URL (must start with https://<language>.wikipedia.org).")
            else:
                try:
                    with st.spinner("Scraping content..."):
                        data = run_shot_scraper(url.strip(), script_path)
                        parsed_content = parse_content(data)
                        html_page = build_html_page(data, parsed_content)
                    st.session_state.parsed_content = parsed_content
                    st.session_state.html_page = html_page
                    st.session_state.scraped_url = url.strip()
                    st.session_state.analysis = None
                except subprocess.CalledProcessError as exc:
                    stderr_text = exc.stderr.strip() if exc.stderr else str(exc)
                    st.error(f"shot-scraper failed: {stderr_text}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

        if st.session_state.parsed_content and st.session_state.scraped_url == url.strip():
            st.subheader("Scraped Content Preview")
            components.html(st.session_state.html_page, height=500, scrolling=True)
        else:
            st.info("Run scrape to fetch and preview page content.")

    with analyze_tab:
        st.header("Analyze content")
        if not (st.session_state.parsed_content and st.session_state.scraped_url == url.strip()):
            st.info("Go to the 'Scrape Page Content' tab and run scraping first.")
            return

        top_n = st.slider("Top N words", min_value=5, max_value=10, value=8)
        analyze_clicked = st.button("Run analysis")

        if analyze_clicked:
            try:
                with st.spinner("Analyzing content..."):
                    unified = unify_content(st.session_state.parsed_content)
                    cleaned = clean_text_content(unified)
                    word_counts = get_word_counts(cleaned)
                    stopword_set = get_stopwords_for_content(url.strip())
                    filtered_counts = filter_by_stopwords(word_counts, stopword_set)
                    top_words = [(w, c) for w, c in filtered_counts.most_common(top_n)]

                st.session_state.analysis = {
                    "unified_length": len(unified),
                    "cleaned_length": len(cleaned),
                    "unique_before": len(word_counts),
                    "unique_after": len(filtered_counts),
                    "top_words": top_words,
                }
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

        if not st.session_state.analysis:
            st.info("Run analysis to compute frequency insights.")
            return

        analysis = st.session_state.analysis
        top_words = analysis["top_words"]     
       
        st.header("Identify Top N Words of Length > 3")
        if not top_words:
            st.warning("No words found after filtering. Try a different page or lower constraints.")
            return

        results_container = st.container()
        with results_container:
            top_words_df = pd.DataFrame(top_words, columns=["word", "count"], index=range(1, len(top_words) + 1))
            st.dataframe(top_words_df, width='stretch')

            words = top_words_df["word"].tolist()
            counts = top_words_df["count"].tolist()

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.bar(words, counts)
                ax.set_xlabel("Words")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Top {len(top_words)} Most Frequent Words (Length > 3)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                wc = WordCloud(width=800, height=500, background_color="white")
                wc = wc.generate_from_frequencies(dict(top_words))
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                ax2.imshow(wc, interpolation="bilinear")
                ax2.axis("off")
                ax2.set_title("Word Cloud")
                plt.tight_layout()
                st.pyplot(fig2)

            st.header("Summarize Insights")
            top_keywords = ", ".join([word for word, _ in top_words[:10]])
            st.write(f"Most frequent keywords in this article include: {top_keywords}.")


if __name__ == "__main__":
    main()