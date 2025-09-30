import aiohttp
import asyncio
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime


class CrawTrendingPapers:
    BASE_URL = "https://huggingface.co"

    async def run(self):
        url = f"{self.BASE_URL}/papers/trending"
        page_content = await self.fetch_url(url)
        paper_cards = await self.parse_trending_cards(page_content)

        papers_data = []
        # 并发请求详情页
        tasks = [self.fetch_url(card["paper_url"]) for card in paper_cards.values()]
        pages = await asyncio.gather(*tasks)

        for (card, page) in zip(paper_cards.values(), pages):
            title, abstract = await self.parse_paper_details(page)

            papers_data.append({
                "title": title,
                "abstract": abstract,
                "url": card["paper_url"],
                "github": card.get("github"),
                "arxiv": card.get("arxiv"),
                "upvotes": card.get("upvotes"),
                "published": card.get("published"),
                "github_stars": card.get("github_stars"),
            })
        return papers_data

    @staticmethod
    async def fetch_url(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

    @staticmethod
    def parse_number(text: str):
        """处理带 k/M 的数字"""
        if not text:
            return None
        text = text.strip().lower().replace(",", "")
        try:
            if text.endswith("k"):
                return int(float(text[:-1]) * 1_000)
            elif text.endswith("m"):
                return int(float(text[:-1]) * 1_000_000)
            elif text.isdigit():
                return int(text)
            return int(text)
        except:
            return None

    @classmethod
    async def parse_trending_cards(cls, page_content):
        """解析 Trending 页面，提取详情页链接 + GitHub + arXiv + Upvote + 日期 + GitHub stars"""
        soup = BeautifulSoup(page_content, "html.parser")
        papers = {}

        for article in soup.find_all("article"):
            a_tag = article.select_one("a[href^='/papers/']")
            if not a_tag:
                continue
            paper_url = f"https://huggingface.co{a_tag['href']}"

            if paper_url not in papers:
                papers[paper_url] = {
                    "paper_url": paper_url,
                    "github": None,
                    "arxiv": None,
                    "upvotes": None,
                    "published": None,
                    "github_stars": None,
                }

            # Upvotes
            upvote_tag = article.select_one(".font-semibold.text-orange-500")
            if upvote_tag:
                papers[paper_url]["upvotes"] = cls.parse_number(upvote_tag.text)

            # Published 日期
            pub_span = article.find("span", string=re.compile(r"Published|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}"))
            if pub_span:
                pub_text = pub_span.text.strip().replace("Published on ", "")
                try:
                    pub_date = datetime.strptime(pub_text, "%b %d, %Y")
                    papers[paper_url]["published"] = pub_date.strftime("%Y-%m-%d")
                except ValueError:
                    papers[paper_url]["published"] = pub_text  # fallback

            # GitHub + stars
            github_link = article.find("a", href=re.compile(r"github.com"))
            if github_link:
                papers[paper_url]["github"] = github_link["href"]
                star_span = github_link.find("span", string=re.compile(r"\d"))
                if star_span:
                    papers[paper_url]["github_stars"] = cls.parse_number(star_span.text)

            # arXiv
            arxiv_link = article.find("a", href=re.compile(r"arxiv.org"))
            if arxiv_link:
                papers[paper_url]["arxiv"] = arxiv_link["href"]

        return papers

    @staticmethod
    async def parse_paper_details(page_content):
        """解析详情页标题和摘要"""
        soup = BeautifulSoup(page_content, "html.parser")
        title_tag = soup.find("h1")
        abstract_tag = soup.find("p")

        title = title_tag.text.strip() if title_tag else "N/A"
        abstract = abstract_tag.text.strip() if abstract_tag else "N/A"
        return title, abstract


# ---------------- 主程序 ----------------
async def main():
    crawler = CrawTrendingPapers()
    papers_data = await crawler.run()

    df = pd.DataFrame(papers_data)
    print(df.head())
    df.to_csv("trending_papers.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
