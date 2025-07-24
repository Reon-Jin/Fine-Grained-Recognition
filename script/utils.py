import numpy as np
import torch


def getROIS(resolution=33, gridSize=3, minSize=1):
    coordsList = []
    step = resolution / gridSize
    for column1 in range(gridSize + 1):
        for column2 in range(gridSize + 1):
            for row1 in range(gridSize + 1):
                for row2 in range(gridSize + 1):
                    x0 = int(column1 * step)
                    x1 = int(column2 * step)
                    y0 = int(row1 * step)
                    y1 = int(row2 * step)
                    if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)):
                        if not (x0 == y0 == 0 and x1 == y1 == resolution):
                            w = x1 - x0
                            h = y1 - y0
                            coordsList.append([x0, y0, w, h])
    return np.array(coordsList)


def getIntegralROIS(resolution=42, step=8, winSize=14):
    coordsList = []
    for column1 in range(0, resolution, step):
        for column2 in range(0, resolution, step):
            for row1 in range(column1 + winSize, resolution + winSize, winSize):
                for row2 in range(column2 + winSize, resolution + winSize, winSize):
                    if row1 > resolution or row2 > resolution:
                        continue
                    x0 = int(column1)
                    y0 = int(column2)
                    x1 = int(row1)
                    y1 = int(row2)
                    if not (x0 == y0 == 0 and x1 == y1 == resolution):
                        w = x1 - x0
                        h = y1 - y0
                        coordsList.append([x0, y0, w, h])
    return np.array(coordsList)


def crop(tensor, dimension, start, end):
    if dimension == 0:
        return tensor[start:end]
    if dimension == 1:
        return tensor[:, start:end]
    if dimension == 2:
        return tensor[:, :, start:end]
    if dimension == 3:
        return tensor[:, :, :, start:end]
    if dimension == 4:
        return tensor[:, :, :, :, start:end]


def squeezefunc(x):
    return torch.squeeze(x, dim=1)


def stackfunc(x):
    return torch.stack(x, dim=1)

