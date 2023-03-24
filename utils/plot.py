import pandas as pd
import json
import mapboxgl
import os
import plotly.express as px


def load_dataset(root, filename):
    full_path = os.path.join(root, filename)
    df = pd.read_csv(full_path, encoding='cp949')
    df.head()
    return df

def load_geodata(root, filename):
    geo = json.load(open(os.path.join(root, filename), encoding='utf-8'))
    return geo

def main():
    root = '../data'
    id_code = '시군구코드'
    value_code = '다이아프램_수리기간'
    name = "충전소_명"
    dataset = load_dataset(root, "hy-station-fault.csv")
    dataset[id_code] = dataset[id_code].astype(str)

    geo_name = "kor-map.geojson"
    geo_data = load_geodata(root, geo_name)
    # set geo data
    for idx, sigun_dict in enumerate(geo_data['features']):
        code = sigun_dict['properties']['SIG_CD']
        count = len(dataset.loc[(dataset[id_code] == code), name])
        value = 0
        tooltip_name = 'None'
        if count > 0:
            value = dataset.loc[(dataset[id_code] == code), value_code].iloc[0]
            tooltip_name = dataset.loc[(dataset[id_code] == code), name].iloc[0]
        else:
            dummy = dataset.iloc[0]
            dummy[id_code] = code
            dummy[value_code] = value
            dataset = dataset.append(dummy)
        geo_data['features'][idx]['properties']['tooltip'] = tooltip_name
        geo_data['features'][idx]['properties']['value'] = value

    # show geo data
    """
    [[0, 'rgb(255,255,255)'],
    [0.1, 'rgb(215,48,39)'],
    [0.2, 'rgb(244,109,67)'],
    [0.3, 'rgb(253,174,97)'],
    [0.4, 'rgb(254,224,144)'],
    [0.5, 'rgb(224,243,248)'],
    [1.0, 'rgb(49,54,149)']],
    """
    fig = px.choropleth_mapbox(dataset, geojson=geo_data, locations='시군구코드', color=value_code,
                               color_continuous_scale=[[0, 'rgb(255,255,255)'],
                                                       [0.1, 'rgb(215,48,39)'],
                                                       [0.2, 'rgb(244,109,67)'],
                                                       [0.3, 'rgb(253,174,97)'],
                                                       [0.4, 'rgb(254,224,144)'],
                                                       [0.5, 'rgb(224,243,248)'],
                                                       [1.0, 'rgb(49,54,149)']],
                               range_color=(0, 10),
                               mapbox_style="carto-positron",
                               featureidkey="properties.SIG_CD", zoom=6, center={"lat":37.565, "lon": 126.986},
                               opacity=0.5, labels={'수소충전소 현황':'지자체별 수소충전소 고장 통게'})
    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
    fig.show()
    test = 1
if __name__ == "__main__":
    main()
