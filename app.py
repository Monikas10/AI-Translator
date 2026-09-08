"""
Flask Web App: AI-Powered Multi-Language Translator
Built with: Flask + LangChain + Google Gemini (gemini-2.5-flash)
"""
from dotenv import load_dotenv
import os
load_dotenv()
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate(
    input_variables=["text", "language", "tone"],
    template="""
    You are a skilled professional translator.
    Translate the following text into {language}.
    Use a {tone} tone/style while preserving the original meaning.
    Only output the translated text, with no extra commentary.

    Text: {text}
    """
)

parser = StrOutputParser()
chain = prompt | llm | parser

TONE_OPTIONS = ["Formal", "Casual / Friendly", "Poetic", "Professional / Business", "Humorous"]

COMMON_LANGUAGES = [
    "Tamil", "English", "Telugu", "Hindi", "Malayalam", "Kannada",
    "French", "Spanish", "German", "Japanese", "Chinese", "Arabic"
]


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    original_text = ""
    selected_tone = "Formal"
    selected_languages = []
    custom_languages = ""

    if request.method == "POST":
        original_text = request.form.get("text", "").strip()
        selected_tone = request.form.get("tone", "Formal")
        selected_languages = request.form.getlist("languages")
        custom_languages = request.form.get("custom_languages", "").strip()

        extra_languages = [lang.strip() for lang in custom_languages.split(",") if lang.strip()]
        languages = selected_languages + extra_languages

        if original_text and languages:
            results = {}
            for lang in languages:
                try:
                    translated = chain.invoke({
                        "text": original_text,
                        "language": lang,
                        "tone": selected_tone
                    })
                    results[lang] = translated.strip()
                except Exception as e:
                    results[lang] = f"Error: {e}"

    return render_template(
        "index.html",
        results=results,
        original_text=original_text,
        selected_tone=selected_tone,
        selected_languages=selected_languages,
        custom_languages=custom_languages,
        tone_options=TONE_OPTIONS,
        common_languages=COMMON_LANGUAGES
    )


if __name__ == "__main__":
    app.run(debug=True)