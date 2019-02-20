from alarm_sound import alarm_sound
from time import sleep
from datetime import datetime, timedelta

ALARM_TIME = input("Alarm when? (default: 10:00)\nHH:MM: ") or "10:00"

now = datetime.now()

hour, minute = map(int, ALARM_TIME.split(":"))
alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

waiting_time = (alarm_time - now).total_seconds() % (24 * 60 * 60)

print(f"Alarm set for {ALARM_TIME}.")
sleep(waiting_time); alarm_sound()