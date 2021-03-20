import json
from datetime import datetime, timezone

import requests

params = {'access_key': '5f249e237bea19f2d36c3adeb55ebd03',
          'output': 'json', 'language': "zh"}
citycode_path = "weather/citycode-2019-08-23.json"
with open(citycode_path) as f:
    citycodes = json.load(f)


def get_weather() -> str:
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
        return str()


def is_night() -> str:
    try:
        city_r = requests.get('http://api.ipstack.com/check', params=params)
        latitude = city_r.json()["latitude"]
        longitude = city_r.json()["longitude"]

        sunset_r = requests.get(
            'https://api.sunrise-sunset.org/json', params={"lat": latitude, "lng": longitude, "formatted": 0})
        sunrise_time = sunset_r.json()['results']['sunrise']
        sunset_time = sunset_r.json()['results']['sunset']
        sunrise_time = datetime.strptime(sunrise_time, "%Y-%m-%dT%H:%M:%S%z")
        sunset_time = datetime.strptime(sunset_time, "%Y-%m-%dT%H:%M:%S%z")

        if sunrise_time < datetime.now(timezone.utc) < sunset_time:
            return False
        else:
            return True
    except:
        return False


if __name__ == "__main__":
    print(get_weather())
    print(is_night())
