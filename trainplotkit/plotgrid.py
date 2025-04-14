from typing import List, Tuple, Mapping, Callable, Any
from torch import nn, Tensor
from torch.utils.data import Dataset
from torcheval.metrics.metric import Metric
import plotly.graph_objects as go
import plotly.callbacks as cb
from plotly.basedatatypes import BaseTraceType
from plotly.subplots import make_subplots
from IPython.display import display
from .utils import to_cpu, to_device
from .layout import place_subplots

class SubPlot():
    ax_type = 'xy'  # https://plotly.com/python/subplots/#subplots-types
    def __init__(self, colspan:int=1, rowspan:int=1):
        self.span = (rowspan, colspan)
        self.parent:"PlotGrid" = None
        self.spi: int = None  # Index within parent's subplot list
        self.position: Tuple[int,int] = None  # (row,col) position in grid

    # Subplot labels and contents
    def title(self) -> str: return ''
    def xlabel(self) -> str: return ''
    def ylabel(self) -> str: return ''

    def create_empty(self, parent, spi, position):
        self.parent = parent
        self.spi = spi
        self.position = position

    # Events
    def before_fit(self): pass
    def after_batch(self, training:bool, inputs:Tensor, targets:Tensor, predictions:Tensor, loss:Tensor): pass
    def after_epoch(self, training:bool): pass
    def after_fit(self): pass
    def on_user_epoch(self, epoch:int): pass
    def on_user_batch(self, batch:int): pass
    def on_user_sample(self, sample:int): pass
    def on_user_channel(self, channel:int): pass
    def before_show_static(self): pass

    # Helpers
    def append_spi(self, name):
        """Ensure that the correct subplot is targeted, e.g. 'xaxis' -> 'xaxis2'"""
        return name if self.spi < 1 else f"{name}{self.spi+1}"
    
    def update_title(self):
        """Convenience method to update the subplot title"""
        self.parent.widget.layout.annotations[self.spi].text = self.title()

    def update_ax_titles(self):
        """Convenience method to update the axis labels of the subplot"""
        xaxis_name = self.append_spi('xaxis')
        yaxis_name = self.append_spi('yaxis')
        kwargs = {xaxis_name: dict(title_text=self.xlabel()),
                  yaxis_name: dict(title_text=self.ylabel())}
        self.parent.widget.update_layout(**kwargs)

    def update_range(self, x_range, y_range):
        """Convenience method to update the axis ranges of the subplot"""
        xaxis_name = self.append_spi('xaxis')
        yaxis_name = self.append_spi('yaxis')
        kwargs = {xaxis_name: dict(range=x_range),
                  yaxis_name: dict(range=y_range)}
        self.parent.widget.update_layout(**kwargs)

    def update_axes(self, xaxis:Mapping, yaxis:Mapping):
        """Convenience method to update custom axis attributes of the subplot"""
        xaxis_name = self.append_spi('xaxis')
        yaxis_name = self.append_spi('yaxis')
        kwargs = {xaxis_name: dict(xaxis),
                  yaxis_name: dict(yaxis)}
        self.parent.widget.update_layout(**kwargs)


class PlotGrid():
    def __init__(self, num_grid_cols:int, subplots:List[SubPlot], fig_height=500):
        self.num_grid_cols = num_grid_cols
        self.subplots = subplots
        self.fig_height = fig_height
        self.widget: go.FigureWidget = None
        self.clicked_trace: BaseTraceType = None
        self.create_empty()

    def show(self):
        """
        Renders the plot in a widget that supports live updates and full user 
        interaction, including click events on traces.

        The widget will not persist beyond the current notebook session.

        This can be called from multiple notebook cells to reduce scrolling 
        fatigue. Under normal circumstances, the resulting widgets will all 
        accept click events and remain synchronized.
        """
        display(self.widget)
        
    def show_static(self, renderer='notebook_connected'):
        """
        Renders the plot in a static figure that will persist beyond the 
        current notebook session.

        The figure does not support live updates and user interaction is 
        limited to the hover, pan and zoom events provided by Plotly.
        """
        for sp in self.subplots: sp.before_show_static()
        spans = [sp.span for sp in self.subplots]
        num_rows, positions, specs, matrix = place_subplots(self.num_grid_cols, spans)
        sp_titles = [sp.title() for sp in self.subplots]
        fig = go.Figure(self.widget)
        fig = make_subplots(rows=num_rows, cols=self.num_grid_cols, specs=specs, subplot_titles=sp_titles, figure=fig)
        fig.show(renderer=renderer)

    def create_empty(self):
        spans = [sp.span for sp in self.subplots]
        num_rows, positions, specs, matrix = place_subplots(self.num_grid_cols, spans)
        sp_titles = [sp.title() for sp in self.subplots]
        self.widget = go.FigureWidget(make_subplots(rows=num_rows, cols=self.num_grid_cols, specs=specs, subplot_titles=sp_titles))
        self.widget.update_layout(height=self.fig_height, margin=dict(l=0, r=0, t=20, b=10))
        for spi, sp in enumerate(self.subplots):
            sp.parent, sp.spi, sp.position = self, spi, positions[spi]
            sp.create_empty(self, spi, positions[spi])
            sp.update_ax_titles()

    def add_trace(self, sp:SubPlot, trace:BaseTraceType): 
        self.widget.add_trace(trace, row=sp.position[0]+1, col=sp.position[1]+1)
        return self.widget.data[-1]  # Object reference to the trace just added
    
    # Register events to trigger if the user clicks on traces 
    def register_user_epoch_event(self, trace:BaseTraceType): 
        trace.on_click(self.on_user_epoch)
    def register_user_batch_event(self, trace:BaseTraceType): 
        trace.on_click(self.on_user_batch)
    def register_user_sample_event(self, trace:BaseTraceType): 
        trace.on_click(self.on_user_sample)
    def register_user_channel_event(self, trace:BaseTraceType): 
        trace.on_click(self.on_user_channel)

    # Events (just forwarded to all subplots)
    def before_fit(self):
        for sp in self.subplots: sp.before_fit()
    def after_batch(self, training:bool, inputs, targets, predictions, loss):
        for sp in self.subplots: sp.after_batch(training, to_cpu(inputs), to_cpu(targets), to_cpu(predictions), to_cpu(loss))
    def after_epoch(self, training:bool):
        for sp in self.subplots: sp.after_epoch(training)
    def after_fit(self):
        for sp in self.subplots: sp.after_fit()
    def on_user_epoch(self, trace, points:cb.Points, selector):
        if not points.point_inds: return
        self.clicked_trace = trace
        epoch = points.point_inds[0]
        for sp in self.subplots: sp.on_user_epoch(epoch)
    def on_user_batch(self, trace, points:cb.Points, selector):
        if not points.point_inds: return
        self.clicked_trace = trace
        batch = points.point_inds[0]
        for sp in self.subplots: sp.on_user_batch(batch)
    def on_user_sample(self, trace, points:cb.Points, selector):
        if not points.point_inds: return
        self.clicked_trace = trace
        sample = points.point_inds[0]
        for sp in self.subplots: sp.on_user_sample(sample)
    def on_user_channel(self, trace, points:cb.Points, selector):
        if not points.point_inds: return
        self.clicked_trace = trace
        channel = points.point_inds[0]
        for sp in self.subplots: sp.on_user_channel(channel)
