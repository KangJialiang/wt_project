def safe_or_not(velocity, distance, weather) -> bool:
    """Call this only when the ultrasonic sensor detects something"""

    safe_flag = True
    possible_weather_conditions = ["小雨", "小到中雨", "中雨", "中到大雨", "大雨",
                                   "大到暴雨", "暴雨", "暴雨到大暴雨", "大暴雨", "大暴雨到特大暴雨", "特大暴雨",
                                   "冻雨", "阵雨", "雷阵雨",
                                   "雨夹雪", "雷阵雨伴有冰雹",
                                   "小雪", "小到中雪", "中雪", "中到大雪", "大雪", "大到暴雪", "暴雪", "阵雪",
                                   "晴", "多云", "阴", "强沙尘暴", "扬沙", "沙尘暴", "浮尘", "雾", "霾"]
    reacting_time = 0.2  # static

    basement_spoting_time = 0.5  # the time before the rider spot the opening door
    basement_acceleration = 8  # m*s^(-2)

    if weather not in possible_weather_conditions:
        weather = "大雨"

    if weather in ["晴", "多云", "阴", "强沙尘暴", "扬沙", "沙尘暴", "浮尘", "雾", "霾"]:
        spoting_time = basement_spoting_time
        acceleration = basement_acceleration
    elif weather in ["小雨", "小到中雨", "中雨", "阵雨"]:
        spoting_time = 0.7
        acceleration = basement_acceleration/1.7
    elif weather in ["中到大雨", "大雨", "雷阵雨"]:
        spoting_time = 0.9
        acceleration = basement_acceleration/1.8
    elif weather in ["大到暴雨", "暴雨", "暴雨到大暴雨", "大暴雨", "大暴雨到特大暴雨", "特大暴雨"]:
        spoting_time = 1
        acceleration = basement_acceleration/2
    elif weather in ["小雪", "小到中雪", "冻雨", "雨夹雪", "阵雪"]:
        spoting_time = 1
        acceleration = basement_acceleration/3
    elif weather in ["雷阵雨伴有冰雹", "中雪", "中到大雪", "大雪", "大到暴雪", "暴雪"]:
        spoting_time = 1
        acceleration = basement_acceleration/4

    if (reacting_time+spoting_time)*velocity+velocity**2/(2*acceleration) >= distance:
        safe_flag = False

    return safe_flag
