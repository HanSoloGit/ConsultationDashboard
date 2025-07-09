import os
import re
import pandas as pd
from docx import Document
from transformers import pipeline
from tqdm import tqdm

# === Load faster zero-shot classification model ===
nli_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",  # much faster than cross-encoder
    device=-1  # -1 = CPU
)

# === Define sentiment labels and templates ===
labels = ["Agree", "Disagree", "Neutral"]
hypothesis_template = {
    "Agree": "The respondent agrees with the following proposal: '{}'",
    "Disagree": "The respondent disagrees with the following proposal: '{}'",
    "Neutral": "The respondent is neutral towards the following proposal: '{}'"
}

# === Extract Q&A blocks ===
def extract_question_answers_with_text(doc_text):
    pattern = r"(.*?)<ESMA_QUESTION_CSDC_(\d+?)>(.*?)<ESMA_QUESTION_CSDC_\2>"
    matches = re.findall(pattern, doc_text, re.DOTALL)
    results = []

    for pre_text, q_number, answer in matches:
        question_lines = [line.strip() for line in pre_text.strip().splitlines() if line.strip()]
        question_text = question_lines[-1] if question_lines else ""
        results.append((q_number, question_text, answer.strip()))
    return results

# === Read .docx text ===
def read_docx_text(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

# === Extract organisation name ===
def extract_organisation_name(doc_text, fallback_name="Unknown"):
    pattern1 = r"(?i)Name of the company\s*/\s*organisation\s*\n(.*?)\n"
    match1 = re.search(pattern1, doc_text)
    if match1:
        return match1.group(1).strip()

    pattern2 = r"We(?:, the)?\s+(.+?),\s+(?:welcome|support|agree|believe|submit)"
    match2 = re.search(pattern2, doc_text)
    if match2:
        return match2.group(1).strip()

    return fallback_name

# === Sentiment classification function ===
def classify_with_nli(answer, question_text):
    hypotheses = [hypothesis_template[label].format(question_text) for label in labels]
    result = nli_classifier(answer, candidate_labels=hypotheses, hypothesis_template="{}")
    label = result["labels"][0]
    score = result["scores"][0]
    final_label = label if score >= 0.7 else "Unclear"

    return {
        "Final Label": final_label,
        "NLI Label": label,
        "NLI Confidence": round(score, 3),
        "All NLI Scores": {lbl: round(scr, 3) for lbl, scr in zip(result["labels"], result["scores"])}
    }

# === Main analysis loop ===
def analyze_folder(folder_path):
    results = []

    file_list = [f for f in os.listdir(folder_path) if f.endswith(".docx")]
    for file_name in tqdm(file_list, desc="🔍 Processing documents"):
        full_path = os.path.join(folder_path, file_name)
        stakeholder = os.path.splitext(file_name)[0]
        doc_text = read_docx_text(full_path)
        organisation_name = extract_organisation_name(doc_text, fallback_name=stakeholder)
        qas = extract_question_answers_with_text(doc_text)

        for q_number, question_text, answer in qas:
            if "TYPE YOUR TEXT HERE" in answer or len(answer.strip()) == 0:
                continue  # skip placeholders and empty answers

            classification = classify_with_nli(answer, question_text)
            results.append({
                "Stakeholder": stakeholder,
                "Organisation": organisation_name,
                "Question ID": f"CSDR_{q_number}",
                "Question Text": question_text,
                "Answer": answer,
                "Answer Length": len(answer),
                "Predicted Label": classification["Final Label"],
                "NLI Label": classification["NLI Label"],
                "NLI Confidence": classification["NLI Confidence"],
                "All NLI Scores": classification["All NLI Scores"]
            })

    return pd.DataFrame(results)

# === Run everything ===
if __name__ == "__main__":
    folder_path = "path"  # ⬅️ Change this path
    df_full = analyze_folder(folder_path)

    # Save full version
    df_full.to_csv("esma_analysis_results_full.csv", index=False)

    # Save dashboard version
    df_dashboard = df_full[[
        "Organisation", "Question ID", "Question Text",
        "Answer", "Predicted Label", "NLI Confidence"
    ]].rename(columns={"NLI Confidence": "Confidence"})

    df_dashboard.to_csv("esma_analysis_results_dashboard_CSDC.csv", index=False)

    print("✅ Done! Results saved as:")
    print("- esma_analysis_results_full.csv")
    print("- esma_analysis_results_dashboard_CSDC.csv")
