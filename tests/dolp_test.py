import time

import dolphin

slider_rl = dolphin.SliderNumber(human_render=False)
slider_rl.auto_run()

print(slider_rl.initial_state)
print(len(slider_rl.initial_state))

slider_rl.set_slider(slider_rl.initial_state)
slider_rl.manual_env.render()
