from nicegui import ui 
import time
import threading

i = 0
started = False
speed = 0.03
range = 5
target = 0

label = ui.label(f'target = {target} to {target + range}')
label = ui.label(f'i = {i}')


slider = ui.slider(min=0, max=100, value=i, step=1).props('disabled')

def move_slider():
    global i
    global started
    started = True
    while started and i < 100:
        i += 1
        slider.value = i
        label.text = f'i = {i}'
        time.sleep(speed)
        if i >= 100:
            i = 0


def start_moving():
    slider.value = 0
    time.sleep(.3)
    threading.Thread(target=move_slider, daemon=True).start()


def stop_moving():
    global started
    global i
    started = False
    i = 0


ui.button('Start', on_click=start_moving)
ui.button('Stop', on_click=stop_moving)

ui.run()