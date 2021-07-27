import matplotlib.pyplot as plt
import numpy as np
import torch
from box import Box

from PGNN.nets.pgnn import PGNN
from PGNN.utils.evaluate_utils import test_over_speed_and_direction

# rc('text', usetex=True)


if __name__ == '__main__':
    # load model from the disk
    config = Box.from_yaml(filename='model_config.yaml')
    model_state_dict = torch.load('model.pt', map_location=torch.device('cpu'))

    m = PGNN(**config.model)
    m.load_state_dict(model_state_dict)
    m.eval()

    wss = np.linspace(6.0, 15.0, 10)
    wds = np.linspace(0, 360, 19)

    ret = test_over_speed_and_direction(m,
                                        config.model_params.use_xy,
                                        config.model_params.use_ws_only,
                                        wss,
                                        wds)

    stack_ret = np.stack(ret)

    # visualize
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(stack_ret, cmap="YlGn")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Test RMSE', rotation=-90, va="bottom")

    ax.set_xticks(np.arange(stack_ret.shape[1]))
    ax.set_xticklabels(wds)
    ax.set_xlabel('Wind direction $(\degree)$')

    ax.set_yticks(np.arange(stack_ret.shape[0]))
    ax.set_yticklabels(wss)
    ax.set_ylabel('Wind speed $(m/s)$')

    _ = plt.setp(ax.get_xticklabels(),
                 rotation=45, ha="right",
                 rotation_mode="anchor")
    plt.show()
