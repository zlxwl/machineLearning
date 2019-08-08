import urllib.request
import ssl
from lxml import etree
import requests
from threading import Thread
from concurrent.futures import ProcessPoolExecutor
import asyncio
import aiohttp


url = 'https://movie.douban.com/top250'
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_1)


# def fetch_page(url):
#     # response = urllib.request.urlopen(url, context=context)
#     response = requests.get(url)
#     return response

async def fetch_content(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


# def fetch_content(url):
#     response = fetch_page(url)
#     page = response.content
#     return page
#
#
# def fetch_page(url):
#     response = requests.get(url)
#     return response

# def parse(url):
#     response = fetch_page(url)
#     # page = response.read()
#     page = response.content
#     html = etree.HTML(page)
#
#     xpath_movie = '//*[@id="content"]/div/div[1]/ol/li'
#     xpath_title = './/span[@class="title"]'
#     xpath_pages = '//*[@id="content"]/div/div[1]/div[2]/a'
#
#     pages = html.xpath(xpath_pages)
#     fetch_list = []
#     result = []
#
#     for element_movie in html.xpath(xpath_movie):
#         result.append(element_movie)
#
#     for p in pages:
#         fetch_list.append(url + p.get('href'))
async def parse(url):
    page = await fetch_content(url)
    html = etree.HTML(page)

    xpath_movie = '//*[@id="content"]/div/div[1]/ol/li'
    xpath_title = './/span[@class="title"]'
    xpath_pages = '//*[@id="content"]/div/div[1]/div[2]/a'

    pages = html.xpath(xpath_pages)
    fetch_list = []
    result = []

    for element_movie in html.xpath(xpath_movie):
        result.append(element_movie)

    for p in pages:
        fetch_list.append(url + p.get('href'))

    task = [fetch_content(url) for url in fetch_list]
    pages = await asyncio.gather(*task)

    for page in pages:
        html = etree.HTML(page)
        for element_movie in html.xpath(xpath_movie):
            result.append(element_movie)


    # #多线程
    # def fetch_content(url):
    #     response = fetch_page(url)
    #     # page = response.read()
    #     page = response.content
    #     html = etree.HTML(page)
    #     for element_movie in html.xpath(xpath_movie):
    #         result.append(element_movie)
    #
    # threads = []
    # for url in fetch_list:
    #     t = Thread(target=fetch_content, args=[url])
    #     t.start()
    #     threads.append(t)
    #
    # for t in threads:
    #     t.join()



    # #多进程
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     for page in executor.map(fetch_content, fetch_list):
    #         html = etree.HTML(page)
    #         for element_movie in html.xpath(xpath_movie):
    #             result.append(element_movie)


    for i, movie in enumerate(result, 1):
        title = movie.find(xpath_title).text
        # print(i, title)


def main():
    loop = asyncio.get_event_loop()
    from time import time
    start = time()
    for i in range(5):
        loop.run_until_complete(parse(url))
    end = time()
    print('cost {} seconds'.format((end - start)/5))
    # parse(url)


if __name__ == '__main__':
    main()
