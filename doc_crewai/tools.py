# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         tools
# Description:  工具
# Author:       shaver
# Date:         2025/5/20
# -------------------------------------------------------------------------------
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from crewai.tools import tool


class Tools:

    @staticmethod
    def scrapeWebsiteTool():
        return ScrapeWebsiteTool(website_url="http://127.0.0.1:8081/openapi.json")

    @tool
    def save_python_to_file(code: str, filename: str) -> None:
        """Useful for when you need to multiply two numbers together."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)

