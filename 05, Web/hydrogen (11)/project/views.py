from django.shortcuts import render
import requests
from bs4 import BeautifulSoup
import pandas as pd

#Create your views here.
def chart(request):
    return render(request,'project/chart.html')

def empty(request):
    return render(request,'project/empty.html')

def form(request):
    return render(request,'project/form.html')

def index(request):
    return render(request,'project/index.html')

def tab_panel(request):
    return render(request,'project/tab-panel.html')

# def table(request):
#     return render(request,'project/table.html')

def ui_elements(request):
    return render(request,'project/ui-elements.html')


## 현범님 크롤링 코드 ##

def news(request):
    response = requests.get('https://search.naver.com/search.naver?where=news&sm=tab_jum&query=%EC%88%98%EC%86%8C+%EC%B6%A9%EC%A0%84')
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    articles = soup.select('#main_pack > section > div > div.group_news > ul > li')
    title_list = []
    url_list = []
    comp_list = []
    thumbnail_list = []
    result = pd.DataFrame(columns=['title','url','com','picture'])

    for article in articles:
        a_tag1 = article.select_one('.news_tit')

        title = a_tag1.text
        title_list.append(title.strip())

        url = a_tag1['href']
        url_list.append(url.strip())

        comp = article.select_one('a.info.press').text
        comp = comp.replace('언론사 선정', '')
        comp_list.append(comp.strip())

        try:
            thumbnail = article.select_one('div > a > img')['src']
            thumbnail_list.append(thumbnail.strip())

        except:
            thumbnail_list.append(
            'https://search.pstatic.net/common/?src=https%3A%2F%2Fimgnews.pstatic.net%2Fimage%2Forigin%2F366%2F2021%2F09%2F23%2F762207.jpg&type=ff264_180&expire=2&refresh=true')

    info = {'title': title_list, 'url': url_list, 'com': comp_list, 'picture': thumbnail_list}

    news = pd.DataFrame(info)
    return render(request, 'project/table.html',{'news' : news})



