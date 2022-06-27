import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.linspace(0, 2 * np.pi, 120)

ims = []
for i in range(60):
    x += np.pi / 15.
    y = np.sin(x)
    im = ax.plot(x, y, animated=True)
    ims.append(im)

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=0)

ani.save("movie.mp4")

plt.show()