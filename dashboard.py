import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import torch
import math

st.set_page_config(page_title="ESMA Consultation Dashboard", layout="wide")
st.title("ESMA Consultation Dashboard")

# === Upload CSV file ===
uploaded_file = st.file_uploader("Upload a CSV file with sentiment analysis", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=';', quoting=1)
    except Exception as e:
        st.error(f"Error loading file: {e}")
    else:
        st.success("File successfully loaded!")

        # === Sidebar filters ===
        with st.sidebar:
            st.header("Filters")

            orgs = sorted(df["Organisation"].dropna().unique())
            select_all_orgs = st.checkbox("Select all organisations", value=True)
            selected_orgs = st.multiselect(
                "Filter by organisation:", orgs, default=orgs if select_all_orgs else []
            )

            question_map = df.groupby("Question ID")["Question Text"].first().to_dict()
            sorted_qids = sorted(question_map.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            question_display = [f"{qid}: {question_map[qid]}" for qid in sorted_qids]
            question_lookup = {f"{qid}: {question_map[qid]}": qid for qid in sorted_qids}

            select_all_questions = st.checkbox("Select all questions", value=False)
            selected_labels = st.multiselect(
                "Filter by question (chronological):",
                question_display,
                default=question_display if select_all_questions else []
            )
            selected_questions = [question_lookup[label] for label in selected_labels]

            keyword = st.text_input("Search keyword in answers:")

            show_sentiment_org = st.checkbox("Show sentiment analysis by organisation", value=False)
            show_sentiment_question = st.checkbox("Show sentiment analysis by question", value=False)
            show_filtered_answers = st.checkbox("Show filtered answers by organisation", value=False)
            show_controversial = st.checkbox("Show most controversial questions", value=False)

        # === Apply filters ===
        filtered_df = df[
            df["Organisation"].isin(selected_orgs) &
            df["Question ID"].isin(selected_questions)
        ]
        if keyword:
            filtered_df = filtered_df[filtered_df["Answer"].str.contains(keyword, case=False, na=False)]

        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
        else:
            # === Normalize Predicted Labels ===
            filtered_df["Predicted Label"] = (
                filtered_df["Predicted Label"]
                .astype(str)
                .apply(lambda x: "Disagree" if "disagrees" in x.lower()
                       else "Agree" if "agrees" in x.lower()
                       else "Neutral" if "neutral" in x.lower()
                       else "Unclear")
            )

            sentiment_colors = {
                "Agree": "green",
                "Disagree": "red",
                "Neutral": "purple",
                "Unclear": "yellow"
            }
            sentiment_order = ["Agree", "Disagree", "Neutral", "Unclear"]

            if len(selected_questions) == 1:
                question_id = selected_questions[0]
                question_text = question_map[question_id]
                st.markdown("---")
                st.markdown(f"### 🧠 Selected Question: {question_id}")
                st.markdown(f"> {question_text}")

            st.markdown("---")
            st.subheader("🔎 Semantic Search over Stakeholder Answers")

            @st.cache_resource
            def load_embedding_model():
                return SentenceTransformer("paraphrase-MiniLM-L6-v2")

            embedding_model = load_embedding_model()

            @st.cache_data
            def compute_embeddings(subset_df):
                return embedding_model.encode(subset_df["Answer"].fillna("").tolist(), convert_to_tensor=True, show_progress_bar=True)

            answer_embeddings = compute_embeddings(filtered_df)

            user_query = st.text_input("Ask your question about stakeholder views:")
            top_k_slider_value = st.slider(
                "Number of top matching answers to show:",
                min_value=1,
                max_value=min(len(filtered_df), 50),
                value=min(5, len(filtered_df))
            )

            if user_query:
                with st.spinner("Searching for relevant answers..."):
                    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
                    cosine_scores = util.pytorch_cos_sim(query_embedding, answer_embeddings)[0]
                    top_k = min(top_k_slider_value, len(filtered_df))
                    top_results = torch.topk(cosine_scores, k=top_k)

                    st.success(f"Top {top_k} matching stakeholder answers:")
                    for score, idx in zip(top_results.values, top_results.indices):
                        row = filtered_df.iloc[idx.item()]
                        st.markdown("---")
                        st.markdown(f"**Organisation:** {row['Organisation']}")
                        st.markdown(f"**Question ID:** {row['Question ID']}")
                        st.markdown(f"**Answer:**  \n> {row['Answer']}")
                        st.markdown(f"_Similarity Score: {score.item():.3f}_")

            if show_filtered_answers:
                st.markdown("---")
                st.subheader("📋 Filtered Answers per Organisation and Question")
                answer_table = filtered_df[[
                    "Organisation", "Question ID", "Question Text",
                    "Answer", "Predicted Label", "Confidence"
                ]].copy()
                answer_table["Question Sort"] = answer_table["Question ID"].apply(
                    lambda x: int(''.join(filter(str.isdigit, x)) or 0)
                )
                answer_table = answer_table.sort_values(["Question Sort", "Organisation"]).drop(columns=["Question Sort"])
                st.dataframe(answer_table, use_container_width=True)
                csv = answer_table.to_csv(index=False).encode("utf-8")
                st.download_button("📅 Download filtered data as CSV", csv, "filtered_answers.csv", "text/csv")

            if show_sentiment_org:
                st.markdown("---")
                st.subheader("📊 Overall Sentiment Count per Organisation")
                fig_org = px.histogram(
                    filtered_df,
                    x="Organisation",
                    color="Predicted Label",
                    barmode="group",
                    title="Overall sentiment distribution by organisation",
                    color_discrete_map=sentiment_colors,
                    category_orders={"Predicted Label": sentiment_order}
                )
                st.plotly_chart(fig_org, use_container_width=True)

            if show_sentiment_question:
                st.markdown("---")
                st.subheader("📊 Overall Sentiment Count per Question (Chronological)")

                filtered_df["Question Sort"] = filtered_df["Question ID"].apply(
                    lambda x: int(''.join(filter(str.isdigit, x)) or 0)
                )
                filtered_df = filtered_df.sort_values("Question Sort")
                all_question_ids = filtered_df["Question ID"].unique().tolist()

                questions_per_chunk = 10
                num_chunks = math.ceil(len(all_question_ids) / questions_per_chunk)

                for i in range(num_chunks):
                    chunk_ids = all_question_ids[i * questions_per_chunk:(i + 1) * questions_per_chunk]
                    chunk_df = filtered_df[filtered_df["Question ID"].isin(chunk_ids)]

                    fig_question = px.histogram(
                        chunk_df,
                        x="Question ID",
                        color="Predicted Label",
                        barmode="group",
                        title=f"Sentiment Distribution – Questions {i * questions_per_chunk + 1} to {(i + 1) * questions_per_chunk}",
                        color_discrete_map=sentiment_colors,
                        category_orders={
                            "Question ID": chunk_ids,
                            "Predicted Label": sentiment_order
                        }
                    )
                    st.plotly_chart(fig_question, use_container_width=True)

            if show_controversial:
                st.markdown("---")
                st.subheader("🔥 Most Controversial Questions (Agree ≈ Disagree)")

                filtered_df = filtered_df[
                    filtered_df["Answer"].notna() &
                    ~filtered_df["Answer"].str.contains("type your text here", case=False, na=False)
                ]

                sentiment_counts = (
                    filtered_df.groupby(["Question ID", "Predicted Label"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=sentiment_order, fill_value=0)
                )

                sentiment_counts["Total Responses"] = sentiment_counts.sum(axis=1)
                min_responses = 5
                valid_counts = sentiment_counts[sentiment_counts["Total Responses"] >= min_responses].copy()

                valid_counts["Controversy Score"] = abs(valid_counts["Agree"] - valid_counts["Disagree"])
                top_n = st.slider("Number of controversial questions to show:", 1, len(valid_counts), 5)

                top_controversial = valid_counts.sort_values("Controversy Score").head(top_n).copy()
                top_controversial = top_controversial.reset_index()
                top_controversial["Question Text"] = top_controversial["Question ID"].map(question_map)
                top_controversial.index += 1  # 1-based ranking

                st.dataframe(top_controversial[[
                    "Question ID", "Question Text", "Agree", "Disagree", "Neutral", "Unclear", "Total Responses", "Controversy Score"
                ]], use_container_width=True)
