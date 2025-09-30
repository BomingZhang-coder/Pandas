from typing import Optional

import aiohttp
import asyncio
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime


def parse_number(text: str) -> Optional[int]:
    """
    解析带 k/M 的数字字符串
    """
    if not text:
        return None
    text = text.lower().strip()
    try:
        if text.endswith("k"):
            return int(float(text[:-1]) * 1000)
        elif text.endswith("m"):
            return int(float(text[:-1]) * 1_000_000)
        else:
            return int(text.replace(",", ""))
    except ValueError:
        return None


class CrawModels:
    BASE_URL = "https://huggingface.co"

    async def run(self):
        url = f"{self.BASE_URL}/models"
        page_content = await self.fetch_url(url)
        models = await self.parse_models(page_content)
        return models

    @staticmethod
    async def fetch_url(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

    @staticmethod
    async def parse_models(page_content):
        soup = BeautifulSoup(page_content, "html.parser")
        models = []

        for article in soup.find_all("article", class_="overview-card-wrapper"):
            a_tag = article.find("a", href=True)
            if not a_tag:
                continue

            # ✅ 只取 h4 里的文本作为 model_id
            h4 = a_tag.find("h4")
            model_id = h4.text.strip() if h4 else a_tag["href"].strip("/")

            model_url = f"https://huggingface.co{a_tag['href']}"

            # ✅ 任务类别（Text Generation）
            task = None
            task_svg = article.find("svg", attrs={"class": re.compile("mr-1.5")})
            if task_svg:
                task = task_svg.find_next(string=True).strip()

            # ✅ 参数量 (22B)
            params = None
            param_span = article.find("span", title=re.compile("Number of parameters"))
            if param_span:
                params = param_span.text.strip()

            # ✅ 更新时间（标准化 YYYY-MM-DD）
            updated = None
            time_tag = article.find("time")
            if time_tag and time_tag.has_attr("datetime"):
                dt = datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00"))
                updated = dt.strftime("%Y-%m-%d")

            # ✅ 下载量 (100k → 100000)
            downloads = None
            download_svg = article.find("svg", attrs={"class": re.compile("w-3 text-gray-400 mr-0.5")})
            if download_svg:
                dl_text = download_svg.find_next(string=True)
                downloads = parse_number(dl_text) if dl_text else None

            # ✅ Likes (665)
            likes = None
            like_svg = article.find("svg", attrs={"class": re.compile("w-3 text-gray-400 mr-1")})
            if like_svg:
                like_text = like_svg.find_next(string=True)
                likes = parse_number(like_text) if like_text else None
                if like_text and like_text.strip().isdigit():
                    likes = int(like_text.strip())

            models.append({
                "model_id": model_id,
                "url": model_url,
                "task": task,
                "parameters": params,
                "updated": updated,
                "downloads": downloads,
                "likes": likes
            })

        return models


# ---------------- 主程序 ----------------
async def main():
    crawler = CrawModels()
    models_data = await crawler.run()

    df = pd.DataFrame(models_data)
    print(df.head())
    df.to_csv("trending_models.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
