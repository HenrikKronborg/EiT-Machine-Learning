from winsound import Beep

def alarm_sound():
    note_frequency = {"F4": 349, "Ab4": 415, "A4": 440, "C5": 523, "E5": 659, "F5": 698}
    
    IMPERIAL_MARCH = [("A4", 500), ("A4", 500), ("A4", 500), ("F4", 350), ("C5", 150), ("A4", 500), ("F4",350), ("C5", 150), ("A4", 1000), ("E5", 500), ("E5", 500), ("E5", 500), ("F5", 350), ("C5", 150), ("Ab4", 500), ("F4", 350), ("C5", 150), ("A4", 1000)]
                      
    for note, duration in IMPERIAL_MARCH:
        Beep(frequency=note_frequency[note], duration=duration)

if __name__ == "__main__":
    alarm_sound()