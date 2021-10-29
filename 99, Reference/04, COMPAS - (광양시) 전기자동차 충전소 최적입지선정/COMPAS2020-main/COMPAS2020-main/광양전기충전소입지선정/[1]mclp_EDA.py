# %%
# Data input
import pathlib
import random
from functools import reduce
from collections import defaultdict

import pandas as pd
import geopandas as gpd
#import folium
import shapely
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import sklearn.cluster
import tensorflow as tf

#from geoband import API

import pydeck as pdk
import os

import pandas as pd


#import cufflinks as cf 
#cf.go_offline(connected=True)
#cf.set_config_file(theme='polar')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Nanum Gothic'

import numpy as np
from shapely.geometry import Polygon, Point
from numpy import random

#최적화 solver
import time
from mip import Model, xsum, maximize, BINARY  
# %%

# 데이터 입력
# 데이터 입력 시 용이성을 위해 df_##으로 이름 지정

df_01= pd.read_csv("../../data/GY/01.광양시_충전기설치현황.csv")
df_02= pd.read_csv("../../data/GY/02.광양시_주차장_공간정보.csv")
df_03= gpd.read_file("../../data/GY/03.광양시_자동차등록현황_격자(100X100).geojson")
df_04= pd.read_csv("../../data/GY/04.광양시_대중집합시설_생활체육시설.csv")
df_05= pd.read_csv("../../data/GY/05.광양시_대중집합시설_야영장.csv")
df_06= pd.read_csv("../../data/GY/06.광양시_전기차보급현황(연도별,읍면동별).csv")
df_07= pd.read_csv("../../data/GY/07.광양시_법정동별_인구현황(읍면동리).csv")
df_08= gpd.read_file("../../data/GY/08.광양시_격자별인구현황(100X100).geojson")
df_09= pd.read_csv("../../data/GY/09.광양시_버스정류장 및 택시승차장_공간정보.csv")
df_10= gpd.read_file("../../data/GY/10.상세도로망.geojson")
df_11= pd.read_csv("../../data/GY/11.평일_일별_시간대별_추정교통량.csv")
df_12= pd.read_csv("../../data/GY/12.평일_전일_혼잡빈도강도.csv")
df_13= pd.read_csv("../../data/GY/13.평일_전일_혼잡시간강도.csv")
df_14= gpd.read_file("../../data/GY/14.광양시_소유지정보.geojson")
df_15= gpd.read_file("../../data/GY/15.광양시_건물정보.geojson")
df_16= gpd.read_file("../../data/GY/16.광양시_도로정보.geojson")
df_17= gpd.read_file("../../data/GY/17.광양시_건물분포도(연면적)_격자(100X100).geojson")
df_18= gpd.read_file("../../data/GY/18.광양시_법정경계(시군구).geojson")
df_19= gpd.read_file("../../data/GY/19.광양시_법정경계(읍면동).geojson")
df_20= gpd.read_file("../../data/GY/20.광양시_행정경계(읍면동).geojson")
df_21= gpd.read_file("../../data/GY/21.광양시_법정경계(리).geojson")
df_22= gpd.read_file("../../data/GY/22.광양시_연속지적.geojson")
df_23= gpd.read_file("../../data/GY/23.광양시_개발행위제한구역.geojson")
df_24= gpd.read_file("../../data/GY/24.광양시_도시계획(공간시설).geojson")
df_25= gpd.read_file("../../data/GY/25.광양시_도시계획(공공문화체육시설).geojson")
df_26= gpd.read_file("../../data/GY/26.광양시_도시계획(교통시설).geojson")
df_27= gpd.read_file("../../data/GY/27.광양시_도시계획(유통공급시설).geojson")
df_28= gpd.read_file("../../data/GY/28.광양시_도시계획(환경기초시설).geojson")
df_29= gpd.read_file("../../data/GY/29.광양시_산업단지(단지경계).geojson")
df_30= gpd.read_file("../../data/GY/30.광양시_산업단지(시설용지도면).geojson")
df_31= gpd.read_file("../../data/GY/31.광양시_산업단지(단지용도지역).geojson")
df_33= gpd.read_file("../../data/GY/33.광양시_고도_격자(100X100).geojson")


# coloumn name 수정
df_11 = df_11.rename(columns = {"상세도로망_링크ID":"link_id"})
df_12 = df_12.rename(columns = {"상세도로망_링크ID":"link_id"})
df_13 = df_13.rename(columns = {"상세도로망 Level6 링크ID":"link_id"})


# %%
#Pydeck 사용을 위한 함수 정의
import geopandas as gpd 
import shapely # Shapely 형태의 데이터를 받아 내부 좌표들을 List안에 반환합니다. 
def line_string_to_coordinates(line_string): 
    if isinstance(line_string, shapely.geometry.linestring.LineString): 
        lon, lat = line_string.xy 
        return [[x, y] for x, y in zip(lon, lat)] 
    elif isinstance(line_string, shapely.geometry.multilinestring.MultiLineString): 
        ret = [] 
        for i in range(len(line_string)): 
            lon, lat = line_string[i].xy 
            for x, y in zip(lon, lat): 
                ret.append([x, y])
        return ret 

def multipolygon_to_coordinates(x): 
    lon, lat = x[0].exterior.xy 
    return [[x, y] for x, y in zip(lon, lat)] 

def polygon_to_coordinates(x): 
    lon, lat = x.exterior.xy 
    return [[x, y] for x, y in zip(lon, lat)] 


# %%
#1. 인구현황 할당
# 격자별 인구 현황
# val 열 na 제거
df_08['val'] = df_08['val'].fillna(0)

# 인구 수 정규화
df_08['정규화인구'] = df_08['val'] / df_08['val'].max()

# geometry를 coordinate 형태로 적용
df_08['coordinates'] = df_08['geometry'].apply(multipolygon_to_coordinates) #pydeck 을 위한 coordinate type

# 100X100 grid에서 central point 찾기
df_08_list = []
df_08_list2 = []
for i in df_08['geometry']:
    cent = [[i[0].centroid.coords[0][0],i[0].centroid.coords[0][1]]]
    df_08_list.append(cent)
    df_08_list2.append(Point(cent[0]))
df_08['coord_cent'] = 0
df_08['geo_cent'] = 0
df_08['coord_cent']= pd.DataFrame(df_08_list) # pydeck을 위한 coordinate type
df_08['geo_cent'] = df_08_list2 # geopandas를 위한 geometry type

# 쉬운 분석을 위한 임의의 grid id 부여
df_08['grid_id']=0
idx = []
for i in range(len(df_08)):
    idx.append(str(i).zfill(5))
df_08['grid_id'] = pd.DataFrame(idx)

# 인구 현황이 가장 높은 위치
df_08.iloc[df_08["val"].sort_values(ascending=False).index].reindex().head()


# 2. 자동차 등록대수
# 격자별 자동차 등록대수

# val 열 na 제거
df_03['totale'].fillna(0)

# coordinate 
df_03['coordinates'] = df_03['geometry'].apply(polygon_to_coordinates) #pydeck 을 위한 coordinate type


# 인구 현황이 가장 높은 위치
df_03.iloc[df_03["totale"].sort_values(ascending=False).index].reindex().head()

#3. 전기자동차
# 전기차 등록 대수 점 수 부여
#년도 별, 행정구역 별, 전기차 보급 추세
list_EV_dist = pd.merge(pd.merge(df_06[df_06["기준년도"]==2017][['행정구역', '보급현황']],                                 
                                 df_06[df_06["기준년도"]==2018][['행정구역', '보급현황']],
                                 how = 'outer', on = '행정구역'),

                                 pd.merge(df_06[df_06["기준년도"]==2019][['행정구역', '보급현황']],
                                 df_06[df_06["기준년도"]==2020][['행정구역', '보급현황']],
                                 how = 'outer', on = '행정구역'),
                                 how = 'outer', on = '행정구역'
                                )

list_EV_dist.columns  = ["ADM_DR_NM", "2017", "2018","2019","2020"]
list_EV_dist=list_EV_dist.iloc[list_EV_dist[["ADM_DR_NM", "2017","2019","2020"]].mean(axis=1).sort_values(ascending=False).index].reindex()
# 2020년 기준으로 가장 많은 비율을 차지하는 광양읍에 전체적으로 점수를 크게 부여할 것
df_EV_ADM = pd.merge(list_EV_dist, df_20, on = "ADM_DR_NM")

#list_EV_dist[["행정구역", "2017","2019","2020"]].mean(axis=1)
df_EV_ADM

#4. 교통량
df = df_10
df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates) 
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()
# 대부분의 사람은 7시에 주거지역에서 업무지역으로 움직일 것으로 가정
# 승용차만 고려
df_11_time7 = df_11[df_11['시간적범위']==7]

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time7[df_11_time7['link_id'].apply(str).str.contains(i)]['승용차'])])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time7=pd.merge(df, df_10_,on = 'link_id' )

# 교통량 합이 가장 높은 도로
df_10_11_time7.iloc[df_10_11_time7["교통량"].sort_values(ascending=False).index].reindex().head()

# 대부분의 사람은 오후 3시에 업무를 하는 것으로 가정 (운송 업 포함)
df_11_time15=df_11[df_11['시간적범위']==15]

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time15[df_11_time15['link_id'].apply(str).str.contains(i)]['승용차'])])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time15=pd.merge(df, df_10_,on = 'link_id' )

# 교통량 합이 가장 높은 도로
df_10_11_time15.iloc[df_10_11_time15["교통량"].sort_values(ascending=False).index].reindex().head()

# 5. 혼잡시간강도

# 혼합빈도강도 양방향 총 합
df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_12[df_12['link_id'].apply(str).str.contains(i)].혼잡빈도강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "혼잡빈도강도합"]
df_10_12=pd.merge(df, df_10_,on = 'link_id' )

# 혼잡빈도강도 합이 가장 높은 도로
df_10_12.iloc[df_10_12["혼잡빈도강도합"].sort_values(ascending=False).index].reindex().head()

# 혼합빈도강도 양방향 총 합

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_13[df_13['link_id'].apply(str).str.contains(i)].혼잡시간강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "혼잡시간강도합"]
df_10_13=pd.merge(df, df_10_,on = 'link_id' )
# 혼잡시간강도 합이 가장 높은 도로
df_10_13.iloc[df_10_13["혼잡시간강도합"].sort_values(ascending=False).index].reindex().head()

# %%
df_14.groupby(['지목코드','지목']).sum()

#%%
# 설치 가능 장소 필터링 (급속 충전소)

df_geo=df_14[df_14['소유구분코드'].isin(['02','04']) 
      & (df_14['지목코드'].isin(['02','05','07','14','15','16','17',
                             '18','19','20','21','27' ,'28'])==False)][['소유구분명','소유구분코드','지목','지목코드','geometry']] # 임야, 염전, 도로, 철도 용지, 제방, 하천 제외 

df_geo['coordinates'] = df_geo['geometry'].apply(multipolygon_to_coordinates) #pydeck 을 위한 coordinate type

# Set the viewport location 
center = [127.696280, 34.940640] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 


layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_geo, # 시각화에 쓰일 데이터프레임 
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Render 
r = pdk.Deck(layers=[layer], initial_view_state=view_state,
             map_style='mapbox://styles/mapbox/outdoors-v11',
             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g"
            )

r.to_html()


# %%
# 설치 가능 장소 필터링 (완속 충전소)

df_geo=df_14[(df_14['지목코드'].isin(['02','05','07','14','15','16','17',
                             '18','19','20','21','27' ,'28'])==False)][['소유구분명','소유구분코드','지목','지목코드','geometry']] # 임야, 염전, 도로, 철도 용지, 제방, 하천 제외 

df_geo['coordinates'] = df_geo['geometry'].apply(multipolygon_to_coordinates) #pydeck 을 위한 coordinate type

# Set the viewport location 
center = [127.696280, 34.940640] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 


layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_geo, # 시각화에 쓰일 데이터프레임 
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Render 
r = pdk.Deck(layers=[layer], initial_view_state=view_state,
             map_style='mapbox://styles/mapbox/outdoors-v11',
             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g"
            )

r.to_html()
# %%
# Assign the value at the standard points

#0. 
df_base = gpd.GeoDataFrame(df_08[['grid_id','val','geometry', 'geo_cent']], geometry = 'geometry')
df_total = df_base
df_base


#1. 교통량
df_superset = df_10_11_time7[df_10_11_time7['교통량']>0]
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry']).buffer(0.001)
df_superset= gpd.GeoDataFrame(df_superset, geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base,df_superset , how = 'right', op = "intersects")
df_superset_result = df_superset_result.groupby('grid_id').sum().reset_index()
df_superset_result = pd.merge(df_superset_result[['grid_id', '교통량']], df_base, on = "grid_id")
df_total = pd.merge(df_total, df_superset_result[['grid_id', '교통량']], how = 'left', on = 'grid_id')
df_total = df_total.rename(columns = {'교통량':'교통량_07'})
df_total['교통량_07'].fillna(0, inplace =True)


df_superset = df_10_11_time7[df_10_11_time15['교통량']>0]
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry']).buffer(0.001)
df_superset= gpd.GeoDataFrame(df_superset, geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base,df_superset , how = 'right', op = "intersects")
df_superset_result = df_superset_result.groupby('grid_id').sum().reset_index()
df_superset_result = pd.merge(df_superset_result[['grid_id', '교통량']], df_base, on = "grid_id")
df_total = pd.merge(df_total, df_superset_result[['grid_id', '교통량']], how = 'left', on = 'grid_id')
df_total = df_total.rename(columns = {'교통량':'교통량_15'})
df_total['교통량_15'].fillna(0, inplace =True)



#2. 혼잡빈도강도
df_superset = df_10_12[df_10_12['혼잡빈도강도합']>0]
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry']).buffer(0.001)
df_superset= gpd.GeoDataFrame(df_superset, geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base,df_superset , how = 'right', op = "intersects")
df_superset_result = df_superset_result.groupby('grid_id').sum().reset_index()
df_superset_result = pd.merge(df_superset_result[['grid_id', '혼잡빈도강도합']], df_base, on = "grid_id")
df_total = pd.merge(df_total, df_superset_result[['grid_id', '혼잡빈도강도합']], how = 'left', on = 'grid_id')
df_total['혼잡빈도강도합'].fillna(0, inplace =True)


#3. 혼잡시간강도
df_superset = df_10_13[df_10_13['혼잡시간강도합']>0]
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry']).buffer(0.001)
df_superset= gpd.GeoDataFrame(df_superset, geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base,df_superset , how = 'right', op = "intersects")
df_superset_result = df_superset_result.groupby('grid_id').sum().reset_index()
df_superset_result = pd.merge(df_superset_result[['grid_id', '혼잡시간강도합']], df_base, on = "grid_id")
df_total = pd.merge(df_total, df_superset_result[['grid_id', '혼잡시간강도합']], how = 'left', on = 'grid_id')
df_total['혼잡시간강도합'].fillna(0, inplace =True)

#4. 자동차 등록대수

df_superset = df_03[df_03['totale']>1].reset_index()
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry'])
df_superset= gpd.GeoDataFrame(df_superset, geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base,df_superset , how = 'right', op = "intersects")
df_superset_result = df_superset_result.groupby('grid_id').sum().reset_index()
df_superset_result = pd.merge(df_superset_result[['grid_id', 'totale']], df_base, on = "grid_id")
df_total = pd.merge(df_total, df_superset_result[['grid_id', 'totale']], how = 'left', on = 'grid_id')
df_total = df_total.rename(columns = {'totale':'자동차등록'})
df_total['자동차등록'].fillna(0, inplace =True)

#5. 전기자동차 등록대수
df_superset = df_EV_ADM[['ADM_DR_CD','2020','geometry']].reset_index()
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry'])
df_superset= gpd.GeoDataFrame(df_superset, geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base,df_superset , how = 'right', op = "intersects")
df_superset_result = df_superset_result.groupby('grid_id').sum().reset_index()
df_superset_result = pd.merge(df_superset_result[['grid_id', '2020']], df_base, on = "grid_id")
df_total = pd.merge(df_total, df_superset_result[['grid_id', '2020']], how = 'left', on = 'grid_id')
df_total = df_total.rename(columns = {'2020':'전기자동차등록'})
df_total['전기자동차등록'].fillna(0, inplace =True)

#6. 기존 충전소
df_01_geo = []
for i in range(len(df_01)):
    df_01_geo.append([df_01.loc[i,'충전소명'],Point(df_01.loc[i,'lon'],df_01.loc[i,'lat']).buffer(0.003)])
#df_01[df_01['급속/완속']=='완속']
df_01_geo = pd.DataFrame(df_01_geo)
df_01_geo.columns = ["충전소명", "geometry"]
df_01_geo = pd.merge(df_01, df_01_geo, on = '충전소명')
df_01_geo['coordinates'] = df_01_geo['geometry'].apply(polygon_to_coordinates) 
df_01_geo = pd.DataFrame(df_01_geo)

df_total['FS_station']=0
df_superset =  df_01_geo[df_01_geo['급속/완속']=='급속'].reset_index()
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry'])
df_superset= gpd.GeoDataFrame(df_superset, geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base,df_superset , how = 'right', op = "intersects").reset_index()
df_superset_result = df_superset_result.groupby('grid_id').sum().reset_index()
location = df_superset_result['grid_id']
for i in location:
    df_total['FS_station'][df_total['grid_id']==i]=1

df_total['SS_station']=0
df_superset =  df_01_geo[df_01_geo['급속/완속']=='완속'].reset_index()
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry'])
df_superset= gpd.GeoDataFrame(df_superset, geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base,df_superset , how = 'right', op = "intersects").reset_index()
df_superset_result = df_superset_result.groupby('grid_id').sum().reset_index()
location = df_superset_result['grid_id']
for i in location:
    df_total['SS_station'][df_total['grid_id']==i]=1

df_base_GEO=gpd.GeoDataFrame(df_base[['grid_id','geo_cent']], geometry ='geo_cent')
#7. 개발 가능
df_14_possible = df_14[df_14['소유구분코드'].isin(['02','04']) 
      & (df_14['지목코드'].isin(['02','05','07','14','15','16','17',
                             '18','19','20','21','27' ,'28'])==False)]
df_superset =  df_14_possible.reset_index()
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry'])
df_superset['geo_Poss_FS'] = 1
df_superset= gpd.GeoDataFrame(df_superset[['geometry','geo_Poss_FS']], geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base_GEO,df_superset , how = 'left', op = "intersects").reset_index()
df_superset_result=df_superset_result[['grid_id','geo_Poss_FS']]
df_superset_result = df_superset_result.fillna(0)
df_superset_result = df_superset_result.groupby(['grid_id']).sum()
df_total= pd.merge(df_total, df_superset_result, on = 'grid_id')


df_14_possible = df_14[(df_14['지목코드'].isin(['02','05','07','14','15','16','17',
                             '18','19','20','21','27' ,'28'])==False)]
df_superset =  df_14_possible.reset_index()
df_superset['geometry'] = gpd.GeoDataFrame(df_superset['geometry'])
df_superset['geo_Poss_SS'] = 1
df_superset= gpd.GeoDataFrame(df_superset[['geometry','geo_Poss_SS']], geometry = 'geometry')
df_superset_result = gpd.sjoin(df_base_GEO,df_superset , how = 'left', op = "intersects").reset_index()
df_superset_result=df_superset_result[['grid_id','geo_Poss_SS']]
df_superset_result = df_superset_result.fillna(0)
df_superset_result = df_superset_result.groupby(['grid_id']).sum()
df_total= pd.merge(df_total, df_superset_result, on = 'grid_id')

df_total[df_total.columns.difference(['geo_cent'])].to_file("df_EDA_result.geojson", driver="GeoJSON")

