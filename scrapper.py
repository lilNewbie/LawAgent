import scrapy


class KanoonSpider(scrapy.Spider):
    name = "kanoon"
    allowed_domains = ["indiankanoon.org"]
    MAX_PAGES = 1

    def __init__(self, search_tags):
        super().__init__()
        self.search_terms = search_tags

    def start_requests(self):
        for query in self.search_terms:
            for page in range(self.MAX_PAGES):
                query_encoded = "+".join(query.split())
                url = f"https://indiankanoon.org/search/?formInput={query_encoded}&pagenum={page}"
                yield scrapy.Request(url, callback=self.parse, meta={"query": query, "page": page})

    def parse(self, response):
        query = response.meta["query"]
        page = response.meta["page"]
        links = response.css("div.result_title a::attr(href)").getall()

        if not links:
            self.logger.info("No results on page %d for query: %s", page, query)
            return

        for href in links:
            yield scrapy.Request(response.urljoin(href), callback=self.process_page)

    def process_page(self, response):
        yield {
            "html": response.text,
            "source_url": response.url,
        }
