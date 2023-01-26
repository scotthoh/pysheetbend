import numpy as np


class GridInfo:
    # grid_shape: np.ndarray(3, dtype=np.float32)
    # grid_start: np.ndarray(3, dtype=np.float32)
    # grid_cell: np.ndarray(3, dtype=np.float32)
    # voxel_size: np.ndarray(3, dtype=np.float32)
    # grid_reci: np.ndarray(3, dtype=np.float32)  # = field(init=False)
    # grid_half: np.ndarray(3, dtype=np.float32)  # = field(init=False)
    # origin: np.ndarray(3, dtype=np.float32)  # = field(init=False)

    def __init__(
        self,
        grid_shape: np.ndarray(3, dtype=np.float32),
        grid_start: np.ndarray(3, dtype=np.float32),
        grid_cell: np.ndarray(3, dtype=np.float32),
        voxel_size: np.ndarray(3, dtype=np.float32),
        origin: np.ndarray(3, dtype=np.float32),
    ):
        self.grid_shape = grid_shape
        self.grid_start = grid_start
        self.grid_cell = grid_cell
        self.voxel_size = voxel_size
        self.grid_reci = np.array(
            (
                self.grid_shape[0],
                self.grid_shape[1],
                self.grid_shape[2] // 2 + 1,
            )
        )
        self.grid_half = np.array(
            (
                self.grid_shape[0] // 2,
                self.grid_shape[1] // 2,
                self.grid_shape[2] // 2,
            )
        )
        self.origin = origin

    def __repr__(self) -> str:
        info = f" INFO: Grid shape : {self.grid_shape}\n"
        info += f" INFO: Voxel size : {self.voxel_size}\n"
        info += f" INFO: Grid start : {self.grid_start}\n"
        info += f" INFO: Origin : {self.origin}\n"

        return info
