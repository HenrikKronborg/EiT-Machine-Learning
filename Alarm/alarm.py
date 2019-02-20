from alarm_sound import alarm_sound
from time import sleep
from datetime import datetime, timedelta

ALARM = "13:45"

now = datetime.now()

hour, minute = map(int, ALARM.split(":"))
alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

waiting_time = (alarm_time - now).total_seconds() % (24 * 60 * 60)

sleep(waiting_time); alarm_sound()