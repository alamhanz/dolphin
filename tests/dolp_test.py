import time

import dolphin

slider_rl = dolphin.SliderNumber(human_render=False)
slider_rl.auto_run()

print(slider_rl.initial_state)
# slider_rl.get_html_template()
time.sleep(5)

print(slider_rl.steps)
