import pywhatkit as kit
import time
import pyautogui

numbers = [
    "no hp"
]

message = ("pesan")

for number in numbers:
    try:
        print(f"Mengirim pesan ke {number}...")
        kit.sendwhatmsg_instantly(number, message, wait_time=10)
        time.sleep(5)
        pyautogui.press("enter")
        print(f"Pesan ke {number} berhasil dikirim.")
        time.sleep(15) 
    except Exception as e:
        print(f"Gagal mengirim ke {number}: {e}")

print("Semua pesan telah dikirim.")
