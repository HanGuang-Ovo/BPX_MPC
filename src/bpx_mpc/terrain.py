"""按世界坐标采样起伏地形，在仿真局部坐标内维持固定大小的高度场。"""
import numpy as np


class RoughTerrain:
    def __init__(self, model, geom):
        self.asset = int(model.geom_dataid[geom])
        rows, cols = int(model.hfield_nrow[self.asset]), int(model.hfield_ncol[self.asset])
        rx, ry = model.hfield_size[self.asset, :2]
        address = model.hfield_adr[self.asset]
        self.samples = model.hfield_data[address:address + rows*cols].reshape(rows, cols)
        self.x_grid = np.linspace(-rx, rx, cols)[None, :]
        self.y_grid = np.linspace(-ry, ry, rows)[:, None]
        # 每次平移三分之一窗口，重叠区域的采样网格保持对齐。
        self.tile_size = np.array([2*rx/3, 2*ry/3])
        self.tile = np.zeros(2, dtype=int)

    @property
    def origin(self):
        return self.tile * self.tile_size

    def offset_tiles(self, xy):
        return np.floor(xy/self.tile_size + .5).astype(int)

    def move(self, tiles):
        self.tile += tiles
        self.update()
        return tiles * self.tile_size

    def update(self):
        origin_x, origin_y = self.origin
        # 每隔 6 m × 4.8 m 布置一组丘包。出生点和丘包之间的通道保持平整。
        x = (self.x_grid + origin_x - 2.45 + 3.) % 6. - 3.
        y = (self.y_grid + origin_y + 2.4) % 4.8 - 2.4
        main = .88 * np.maximum(0., 1. - ((x+.32)/1.4)**2 - ((y+.10)/.90)**2)**2
        side = .75 * np.maximum(0., 1. - ((x-.50)/1.15)**2 - ((y-.50)/.88)**2)**2
        self.samples[:] = main + side*(1.-main)
