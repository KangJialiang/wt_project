import json

import requests

params = {'access_key': '5f249e237bea19f2d36c3adeb55ebd03',
          'output': 'json', 'language': "zh"}
citycode_path = "weather/citycode-2019-08-23.json"
with open(citycode_path) as f:
    citycodes = json.load(f)


def get_weather():
    try:
        city_r = requests.get('http://api.ipstack.com/check', params=params)
        city = city_r.json()["city"]
        for it in citycodes:
            if city in it["city_name"] or it["city_name"] in city:
                city_code = it["city_code"]
                weather_r = requests.get(
                    f"http://t.weather.itboy.net/api/weather/city/{city_code}")
                weather = weather_r.json()["data"]["forecast"][0]["type"]
                return weather
        else:
            raise ValueError
    except:
        return None


if __name__ == "__main__":
    print(get_weather())
