"""Export the exact excluded crash cohort and a standalone interactive review map."""
from pathlib import Path
import json,datetime,sys,tempfile,urllib.request
import duckdb,pandas as pd,h3
ROOT=Path(__file__).resolve().parents[1]
BOXES={'saratov':(51.4,45.8,51.7,46.2),'samara':(53.1,49.95,53.35,50.4),'ulyanovsk':(54.15,48.15,54.45,48.65)}
NAMES={'saratov':'Саратов','samara':'Самара','ulyanovsk':'Ульяновск'}
def inside(lat,lon,b):return b[0]<=lat<=b[2] and b[1]<=lon<=b[3]
def main():
    dest=Path.home()/'Downloads'/'ДТП_вне_дорожных_ячеек_2015–2025'
    dest.mkdir(exist_ok=True)
    c=duckdb.connect()
    q=c.execute("""SELECT *,year(dt) AS year FROM (SELECT crash_uid,EM_NUMBER,try_cast(DATE_TIME AS TIMESTAMP) AS dt,
        coalesce(POINT_LAT,LNG) AS lat,coalesce(POINT_LNG,LAT) AS lon,EM_TYPE,NP,STREET,HOUSE,PLACE,ROADS,ROAD_KM_M,
        LOST_AMOUNT,SUFFER_AMOUNT FROM read_parquet(?))
        WHERE year(dt) BETWEEN 2015 AND 2025 AND lat BETWEEN 51.4 AND 54.45 AND lon BETWEEN 45.8 AND 50.4""",[str(ROOT/'parquet_output/crashes.parquet')]).fetchdf()
    allrows=[];audit=[]
    ref=pd.read_csv(ROOT/'docs/analysis/transfer_eda/coordinate_coverage_by_year.csv')
    for city,b in BOXES.items():
        if city=='saratov':cells=set(pd.read_parquet(ROOT/'parquet_output/h3_year_panel.parquet',columns=['h3']).h3)
        else:cells=set(pd.read_parquet(ROOT/f'reports/transfer_cities/{city}_features.parquet',columns=['h3']).h3)
        z=q[q.lat.between(b[0],b[2])&q.lon.between(b[1],b[3])].copy();z['h3']=[h3.latlng_to_cell(a,b,9) for a,b in zip(z.lat,z.lon)];z=z[~z.h3.isin(cells)].copy()
        z['city']=city;z['reason']=['Центр ячейки за границей окна' if not inside(*h3.cell_to_latlng(h),b) and city!='saratov' else 'Ячейка отсутствует в дорожной выборке' for h in z.h3]
        for y in range(2015,2026):
            n=int((z.year==y).sum());expected=int(ref[(ref.city==city)&(ref.year==y)].excluded.iloc[0]);assert n==expected,(city,y,n,expected)
            audit.append({'Город':NAMES[city],'Год':y,'Число ДТП':n})
        allrows.append(z)
    df=pd.concat(allrows).sort_values(['city','dt','crash_uid']).reset_index(drop=True)
    df['coord_n']=df.groupby(['city','lat','lon']).crash_uid.transform('size')
    df['maps_url']=['https://yandex.ru/maps/?ll='+str(lon)+'%2C'+str(lat)+'&z=18&pt='+str(lon)+'%2C'+str(lat)+'%2Cpm2rdm' for lat,lon in zip(df.lat,df.lon)]
    cols={'city':'Город','crash_uid':'ID строки в исходном наборе','EM_NUMBER':'Номер ДТП','dt':'Дата и время','year':'Год','lat':'Широта','lon':'Долгота','h3':'H3 ячейка (res 9)','EM_TYPE':'Вид ДТП','NP':'Населённый пункт в карточке','STREET':'Улица','HOUSE':'Дом','PLACE':'Место','ROADS':'Дорога','ROAD_KM_M':'Километр дороги','LOST_AMOUNT':'Погибшие','SUFFER_AMOUNT':'Раненые','reason':'Причина исключения','coord_n':'ДТП с теми же координатами','maps_url':'Открыть в Яндекс Картах'}
    export=df[list(cols)].rename(columns=cols);export['Город']=export['Город'].map(NAMES)
    export.to_csv(dest/'ДТП_список.csv',sep=';',encoding='utf-8-sig',index=False)
    pd.DataFrame(audit).to_csv(dest/'Сводка_по_годам.csv',sep=';',encoding='utf-8-sig',index=False)
    data=[];geo=[]
    for r in df.to_dict('records'):
        clean={k:(None if pd.isna(v) else str(v) if isinstance(v,(datetime.datetime,pd.Timestamp)) else v) for k,v in r.items()}
        data.append(clean);geo.append({'type':'Feature','geometry':{'type':'Point','coordinates':[r['lon'],r['lat']]},'properties':clean})
    (dest/'ДТП_точки.geojson').write_text(json.dumps({'type':'FeatureCollection','features':geo},ensure_ascii=False))
    # Nearby contemporary roads are for visual inspection, not proof of historical road presence.
    occupied={(round(r['lat']*100),round(r['lon']*100)) for r in data};neighbor={(a+i,b+j) for a,b in occupied for i in (-1,0,1) for j in (-1,0,1)}
    roads=[];raw=pd.read_parquet(ROOT/'parquet_output/osm_roads_raw.parquet',columns=['osm_id','highway','name','coords_json'])
    cache=json.loads((ROOT/'reports/transfer_cities/extracted_roads.json').read_text())
    candidates=[(str(r.osm_id),r.highway,str(r.name or ''),json.loads(r.coords_json)) for r in raw.itertuples()]
    for city,d in cache.items():
        nodes=d['nodes']
        for w in d['ways']:
            coords=[[nodes[str(n)]['lat'],nodes[str(n)]['lon']] for n in w['nodes']]
            candidates.append((str(w['id']),w['tags'].get('highway',''),w['tags'].get('name',''),coords))
    seen=set()
    for osm_id,kind,name,coords in candidates:
        if osm_id in seen:continue
        if any((round(a*100),round(b*100)) in neighbor for a,b in coords):roads.append({'id':osm_id,'kind':kind,'name':name,'coords':coords});seen.add(osm_id)
    template=(ROOT/'scripts/excluded_crashes_map.html').read_text()
    assets=Path(tempfile.gettempdir())/'dtp-map-assets';assets.mkdir(exist_ok=True)
    for filename in ['leaflet.css','leaflet.js']:
        asset=assets/filename
        if not asset.exists():
            with urllib.request.urlopen('https://unpkg.com/leaflet@1.9.4/dist/'+filename,timeout=60) as response:
                asset.write_bytes(response.read())
    css=(assets/'leaflet.css').read_text();js=(assets/'leaflet.js').read_text()
    bounds={k:[[b[0],b[1]],[b[2],b[3]]] for k,b in BOXES.items()}
    polygons=[{'h3':cell,'coords':h3.cell_to_boundary(cell)} for cell in sorted(set(df.h3))]
    page=template.replace('__LEAFLET_CSS__',css).replace('__LEAFLET_JS__',js).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')).replace('__ROADS__',json.dumps(roads,ensure_ascii=False).replace('</','<\\/')).replace('__CELLS__',json.dumps(polygons)).replace('__BOXES__',json.dumps(bounds))
    (dest/'Карта_ДТП.html').write_text(page)
    (dest/'Прочитай_меня.txt').write_text('ДТП вне дорожных H3-ячеек: исследуемые окна Саратова, Самары и Ульяновска, 2015–2025.\nНе вся Россия и не административные границы городов.\nCSV: UTF-8 с BOM, разделитель точка с запятой. Координаты WGS84. ID строки — внутренний crash_uid; Номер ДТП — EM_NUMBER исходной карточки.\nШирота и долгота восстановлены тем же правилом, что в crashes_slim: POINT_LAT/POINT_LNG, при отсутствии — LNG/LAT соответственно.\nПричина отсутствия в дорожной выборке не установлена: возможны неточности координат, изменения сети, неполнота OSM. Граничные ячейки выделены отдельной причиной.\nКарта содержит точки и ближайшие дороги из локального снимка OSM июня 2026. Картографическая и спутниковая подложки требуют интернета и имеют свои даты обновления. Дороги снимка можно включить отдельно.\nВ карте можно фильтровать годы, города, причины; выделить прямоугольник двумя щелчками и скачать CSV выбранных точек. CSV выборки сохраняет все ДТП группы координат.\nЗначок с числом означает несколько ДТП с одинаковыми координатами.\n\nВсего ДТП: '+str(len(df))+'\n'+df.groupby('city').size().rename(index=NAMES).to_string()+'\n')
    assert len(df)==df.crash_uid.nunique()
    assert pd.read_csv(dest/'ДТП_список.csv',sep=';').shape[0]==len(df)
    print(dest);print(df.groupby('city').size());print('total',len(df),'unique coordinates',df.groupby(['city','lat','lon']).ngroups,'roads',len(roads))
if __name__=='__main__':main()
