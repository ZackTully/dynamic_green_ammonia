import numpy as np


def moving_average(data, width):
    if width <= 1:
        return data
    cs = np.cumsum(data)
    filtered_data = (cs[width:] - cs[:-width]) / width
    std_dev = np.zeros(len(filtered_data))
    for i in range(len(std_dev)):
        std_dev[i] = np.std(data[i : i + width])

    x = np.arange(0, len(data), 1)
    xp = np.linspace(0, len(data), int(len(data) - width))
    filtered_data = np.interp(x, xp, filtered_data)
    std_dev = np.interp(x, xp, std_dev)

    # filtered_data = np.interp(
    #     np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), filtered_data
    # )
    # std_dev = np.interp(
    #     np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), std_dev
    # )

    return filtered_data, std_dev


class CommonCalcs:
    def __init__(self):
        pass
