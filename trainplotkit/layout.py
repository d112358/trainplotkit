from typing import List, Tuple, Mapping, Any
import torch
from torch import Tensor

def calc_specs(num_cols:int, num_rows:int, positions:List[Tuple[int, int]], spans:List[Tuple[int, int]]):
    """
    Determine `specs` parameter to pass to `make_subplots` based on grid 
    top-level corner `positions` and `spans` of subplots
    """
    specs = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for pos,span in zip(positions,spans):
        specs[pos[0]][pos[1]] = dict(rowspan=span[0], colspan=span[1])
    return specs

def grid_cell_gen(num_cols):
    """Generator of (row,col) tuples for a grid of specified width"""
    row = 0
    while True:
        for col in range(num_cols): yield (row, col)
        row += 1

def ensure_matrix_height(matrix:Tensor, pos:Tuple[int,int], span:Tuple[int,int]=(1,1)) -> Tensor:
    """
    Checks if the matrix has enough rows to evaluate `pos[0]+span[0]`. Expands 
    the matrix if required.
    """
    num_grid_cols = matrix.shape[1]
    required_rows = pos[0] + span[0] - matrix.shape[0]
    if required_rows > 0:
        new_rows = torch.full((required_rows, num_grid_cols), fill_value=-1)
        matrix = torch.cat((matrix, new_rows), dim=0)
    return matrix

def cells_available(matrix:Tensor, pos:Tuple[int,int], span:Tuple[int,int]=(1,1)) -> Tuple[bool, Tensor]:
    """
    Checks whether a subplot with top left corner at `pos` and extent of 
    `span` will fit into `matrix` without exceeding the matrix width or 
    overlapping with subplots already placed. Automatically extends the 
    matrix height if required.
    """
    # Check width and height
    matrix = ensure_matrix_height(matrix, pos, span)
    if span[1] > matrix.shape[1]: raise ValueError(f'Subplot ({span[1]}) is wider than grid ({matrix.shape[1]})')
    if pos[1] + span[1] > matrix.shape[1]: return False, matrix  # Off the right-most edge

    # Check cell contents
    target_cells = matrix[pos[0]:pos[0]+span[0], pos[1]:pos[1]+span[1]]
    is_available = bool((target_cells < 0).all())
    return is_available, matrix

def place_subplots(num_grid_cols:int, spans:List[Tuple[int, int]]) -> Tuple[int, List[Tuple[int,int]], List[List[Mapping]], Tensor]:
    """
    Calculate subplot placement parameters based on the number of columns in 
    the grid and the column and row spans of each subplot.

    Input parameters:
    * `num_grid_cols` is the number of subplots with a column span of 1 that 
      can fit into a single row. The number of columns in the grid is fixed 
      beforehand, whereas the number of rows grow as required.
    * spans is a list of `(rowspan,colspan)` tuples specifying the span of 
      each subplot.
    
    Output parameters:
    * `num_grid_rows` is the calculated number of rows required to fit all the 
      subplots
    * `positions` is a list of `(row,col)` tuples containing the calculated 
      positions of the top-left corners of each subplot. These are 0-based 
      and must be incremented by 1 when used as inputs to 
      `FigureWidget.add_trace`
    * `specs` is a 2D nested list of dictionaries that can be passed to 
      `make_subplots` as described here:
      https://plotly.com/python/subplots/#multiple-custom-sized-subplots
    * `matrix` is a 2D tensor in which each cell contains the index of the 
      subplot that will occupy the corresponding cell in the grid. Unoccupied 
      cells have the value -1.
    """
    matrix = torch.empty(0, num_grid_cols)
    positions:List[Tuple[int,int]] = []
    cell_gen = grid_cell_gen(num_grid_cols)
    cur_pos = next(cell_gen)
    for spi, span in enumerate(spans):
        is_available, matrix = cells_available(matrix, cur_pos, span)
        while not is_available: 
            cur_pos = next(cell_gen)
            is_available, matrix = cells_available(matrix, cur_pos, span)
        matrix[cur_pos[0]:cur_pos[0]+span[0], cur_pos[1]:cur_pos[1]+span[1]] = spi
        positions.append(cur_pos)
    num_grid_rows = matrix.shape[0]
    specs = calc_specs(num_grid_cols, num_grid_rows, positions, spans)
    return num_grid_rows, positions, specs, matrix
