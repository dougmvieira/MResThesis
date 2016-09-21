def days_to_seconds(days):
    seconds_in_a_trading_day = (9*60 + 30)*60
    return days*seconds_in_a_trading_day


def miliseconds_to_days(ms):
    minutes_in_a_trading_day = 9*60 + 30
    miliseconds_in_a_minute = 60*1000
    return ms/(minutes_in_a_trading_day*miliseconds_in_a_minute)
