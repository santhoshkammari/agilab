def content_extractor_parse_urls(urls):
    from liteauto.parselite import parse
    result = parse(urls)
    return result


def content_extractor_html_to_markdown(html):
    from pyhtml2md import convert
    return convert(html) if html else ""


def content_extractor_batch_process(urls):
    htmls = content_extractor_parse_urls(urls)
    results = {}
    for x in htmls:
        results[x.url] = content_extractor_html_to_markdown(x.content)
    return results