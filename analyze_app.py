import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import asyncio

# 导入两个爬虫类
from crawl_paper import CrawTrendingPapers
from crawl_models import CrawModels

# ==============================
# 1. 数据获取
# ==============================
@st.cache_data(show_spinner="正在爬取最新 HuggingFace 数据…")
def load_data():
    """运行爬虫，返回 DataFrame"""
    async def run_crawlers():
        # 爬取 papers
        paper_crawler = CrawTrendingPapers()
        papers_data = await paper_crawler.run()
        papers_df = pd.DataFrame(papers_data)

        # 爬取 models
        model_crawler = CrawModels()
        models_data = await model_crawler.run()
        models_df = pd.DataFrame(models_data)

        return papers_df, models_df

    return asyncio.run(run_crawlers())

# ==============================
# 2. 文本处理工具
# ==============================
def generate_wordcloud(text_series, title="Word Cloud"):
    text = " ".join(str(t) for t in text_series.dropna())
    if not text.strip():
        st.info("暂无文本数据生成词云")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)

# ==============================
# 3. Streamlit 主界面
# ==============================
def main():
    st.set_page_config(page_title="AI Trending Dashboard", layout="wide")
    st.title("🤖 AI Trending Papers & Models Dashboard")
    st.write("数据来源：[Hugging Face Trending Papers](https://huggingface.co/papers/trending) "
             "与 [Hugging Face Trending Models](https://huggingface.co/models)")

    # 获取数据
    papers_df, models_df = load_data()

    # 数值处理
    for col in ["upvotes", "github_stars"]:
        if col in papers_df.columns:
            papers_df[col] = pd.to_numeric(papers_df[col], errors="coerce")

    for col in ["downloads", "likes"]:
        if col in models_df.columns:
            models_df[col] = pd.to_numeric(models_df[col], errors="coerce")

    if "published" in papers_df.columns:
        papers_df["published"] = pd.to_datetime(papers_df["published"], errors="coerce")
    if "updated" in models_df.columns:
        models_df["updated"] = pd.to_datetime(models_df["updated"], errors="coerce")

    # ================= 全局指标 =================
    st.subheader("🌍 全局指标")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("论文总数", len(papers_df))
    col2.metric("总 Upvotes", int(papers_df["upvotes"].sum(skipna=True)))
    col3.metric("模型总数", len(models_df))
    col4.metric("总 Downloads", int(models_df["downloads"].sum(skipna=True)))

    # ================= Tabs =================
    tab1, tab2, tab3 = st.tabs(["📄 Papers", "🧩 Models", "📊 对比分析"])

    # ---------------- Papers ----------------
    with tab1:
        st.header("Trending Papers 分析")

        # 搜索框
        query = st.text_input("🔍 搜索论文 (按标题/摘要/URL)", "")
        filtered_papers = papers_df.copy()
        if query:
            mask = (
                    papers_df["title"].str.contains(query, case=False, na=False) |
                    papers_df["abstract"].str.contains(query, case=False, na=False) |
                    papers_df["url"].str.contains(query, case=False, na=False)
            )
            filtered_papers = papers_df[mask]

        st.dataframe(filtered_papers.head(50))

        st.subheader("🔥 Top 10 Papers by Upvotes")
        top_papers = papers_df.sort_values("upvotes", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=top_papers, x="upvotes", y="title", palette="Blues_r", ax=ax)
        st.pyplot(fig)

        if "published" in papers_df.columns:
            st.subheader("📅 论文发布时间趋势")
            papers_df["year_month"] = papers_df["published"].dt.to_period("M")
            pub_trend = papers_df.groupby("year_month").size()
            st.line_chart(pub_trend)

        st.subheader("📖 摘要关键词 Word Cloud")
        generate_wordcloud(papers_df["abstract"], "Papers Abstracts")

        if "github_stars" in papers_df.columns:
            st.subheader("⭐ Upvotes vs GitHub Stars")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=papers_df, x="upvotes", y="github_stars", alpha=0.7, ax=ax)
            ax.set_xscale("log")
            ax.set_yscale("log")
            st.pyplot(fig)

    # ---------------- Models ----------------
    with tab2:
        st.header("Trending Models 分析")
        # 搜索框
        query = st.text_input("🔍 搜索模型 (按ID/任务)", "")
        filtered_models = models_df.copy()
        if query:
            mask = (
                    models_df["model_id"].str.contains(query, case=False, na=False) |
                    models_df["task"].str.contains(query, case=False, na=False)
            )
            filtered_models = models_df[mask]

        st.dataframe(filtered_models.head(30))

        st.subheader("⬇️ Top 10 Models by Downloads")
        top_models = models_df.sort_values("downloads", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=top_models, x="downloads", y="model_id", palette="Greens_r", ax=ax)
        st.pyplot(fig)

        st.subheader("💡 模型 Likes vs Downloads")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=models_df, x="downloads", y="likes", hue="task", alpha=0.7, ax=ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        st.pyplot(fig)

        st.subheader("🧩 模型任务分布")
        st.bar_chart(models_df["task"].value_counts().head(10))

        if "parameters" in models_df.columns:
            st.subheader("📏 模型参数量分布（Parameters）")
            fig, ax = plt.subplots(figsize=(8, 6))
            models_df["parameters"].dropna().value_counts().head(15).plot(kind="barh", ax=ax, color="orange")
            st.pyplot(fig)

        st.subheader("📖 模型任务关键词 Word Cloud")
        generate_wordcloud(models_df["task"].fillna("Unknown"), "Model Tasks")

    # ---------------- 对比 ----------------
    with tab3:
        st.header("📊 Papers vs Models 对比分析")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("论文平均 Upvotes", round(papers_df["upvotes"].mean(skipna=True), 2))
            st.metric("论文平均 GitHub Stars", round(papers_df["github_stars"].mean(skipna=True), 2))
        with col2:
            st.metric("模型平均 Downloads", round(models_df["downloads"].mean(skipna=True), 2))
            st.metric("模型平均 Likes", round(models_df["likes"].mean(skipna=True), 2))

        if "published" in papers_df.columns and "updated" in models_df.columns:
            st.subheader("📅 Papers vs Models 时间趋势")
            papers_df["year_month"] = papers_df["published"].dt.to_period("M")
            models_df["year_month"] = models_df["updated"].dt.to_period("M")

            compare_df = pd.DataFrame({
                "Papers": papers_df.groupby("year_month").size(),
                "Models": models_df.groupby("year_month").size()
            }).fillna(0)
            st.line_chart(compare_df)

        st.subheader("🔍 主题对比：Papers Abstract vs Models Tasks")
        col1, col2 = st.columns(2)
        with col1:
            generate_wordcloud(papers_df["abstract"], "Papers Abstracts")
        with col2:
            generate_wordcloud(models_df["task"].fillna("Unknown"), "Models Tasks")


if __name__ == "__main__":
    main()
