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
        self.spi: int = None  # Index within parent's sub-plot list
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
    def after_batch(self, training:bool, inputs:Tensor, targets:Tensor, predictions:Tensor, loss:Tensor): pass
    def after_epoch(self, training:bool): pass
    def after_fit(self): pass
    def on_user_epoch(self, epoch:int): pass
    def on_user_sample(self, sample:int): pass
    def on_user_channel(self, channel:int): pass

    # Helpers
    def append_spi(self, name):
        """Ensure that the correct sub-plot is targeted, e.g. 'xaxis' -> 'xaxis2'"""
        return name if self.spi < 1 else f"{name}{self.spi+1}"
    
    def update_ax_titles(self):
        """Convenience method to update the axis labels of the sub-plot"""
        xaxis_name = self.append_spi('xaxis')
        yaxis_name = self.append_spi('yaxis')
        kwargs = {xaxis_name: dict(title_text=self.xlabel()),
                  yaxis_name: dict(title_text=self.ylabel())}
        self.parent.widget.update_layout(**kwargs)

    def update_range(self, x_range, y_range):
        """Convenience method to update the axis ranges of the sub-plot"""
        xaxis_name = self.append_spi('xaxis')
        yaxis_name = self.append_spi('yaxis')
        kwargs = {xaxis_name: dict(range=x_range),
                  yaxis_name: dict(range=y_range)}
        self.parent.widget.update_layout(**kwargs)

    def update_axes(self, xaxis:Mapping, yaxis:Mapping):
        """Convenience method to update custom axis attributes of the sub-plot"""
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
        self.widget.show(renderer=renderer)

    def create_empty(self):
        spans = [sp.span for sp in self.subplots]
        num_rows, positions, specs, matrix = place_subplots(self.num_grid_cols, spans)
        sp_titles = [sp.title() for sp in self.subplots]
        self.widget = go.FigureWidget(make_subplots(rows=num_rows, cols=self.num_grid_cols, specs=specs, subplot_titles=sp_titles))
        self.widget.update_layout(height=self.fig_height)
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
    def register_user_sample_event(self, trace:BaseTraceType): 
        trace.on_click(self.on_user_sample)
    def register_user_channel_event(self, trace:BaseTraceType): 
        trace.on_click(self.on_user_channel)

    # Events (just forwarded to all sub-plots)
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


class AxisRange():
    """Convenience class to simplify keeping ranges updated with plot contents"""
    def __init__(self, min_x=0, max_x=1, min_y=0, max_y=1):
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y

    def x_range(self): return [self.min_x, self.max_x]
    def y_range(self): return [self.min_y, self.max_y]

    def update(self, x_values, y_values):
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        range_changed = False
        if min_x < self.min_x: self.min_x = min_x; range_changed = True
        if max_x > self.max_x: self.max_x = max_x; range_changed = True
        if min_y < self.min_y: self.min_y = min_y; range_changed = True
        if max_y > self.max_y: self.max_y = max_y; range_changed = True
        return range_changed

class TrainingCurveSP(SubPlot):
    """Plots the training and validation loss as a function of epoch"""
    def __init__(self, colspan:int=1, rowspan:int=1):
        super().__init__(colspan, rowspan)
        self.xy_range = AxisRange(0, 10, 0, 0.001)
        self.train_loss_trace: BaseTraceType = None
        self.valid_loss_trace: BaseTraceType = None
        self.marker_trace: BaseTraceType = None
        self.epoch = 0
        self.train_num = 0
        self.train_denom = 0
        self.valid_num = 0
        self.valid_denom = 0
        
    def title(self) -> str: return 'Training curve'
    def xlabel(self) -> str: return 'Epoch'
    def ylabel(self) -> str: return 'Loss'

    def create_empty(self, parent:PlotGrid, spi, position):
        super().create_empty(parent, spi, position)
        train_loss_trace = go.Scatter(x=[], y=[], mode='lines+markers', name='Training loss')
        valid_loss_trace = go.Scatter(x=[], y=[], mode='lines+markers', name='Validation loss')
        marker_trace     = go.Scatter(x=[], y=[], mode='markers', showlegend=False, hoverinfo='skip',
                                      marker=dict(color='rgba(0,0,0,0.2)', line=dict(color='black', width=2)))
        
        self.train_loss_trace = parent.add_trace(self, train_loss_trace)
        self.valid_loss_trace = parent.add_trace(self, valid_loss_trace)
        self.marker_trace     = parent.add_trace(self, marker_trace)
    
    def after_batch(self, training, inputs, targets, predictions, loss):
        if training:
            self.train_num += float(loss.detach().cpu())
            self.train_denom += 1
        else:
            self.valid_num += float(loss.detach().cpu())
            self.valid_denom += 1

    def after_epoch(self, training):
        if training:
            loss = self.train_num / self.train_denom
            new_x = tuple(self.train_loss_trace.x) + (self.epoch,)
            new_y = tuple(self.train_loss_trace.y) + (loss,)
            self.train_loss_trace.update(x=new_x, y=new_y)
            self.train_num = 0
            self.train_denom = 0
        else:
            loss = self.valid_num / self.valid_denom
            new_x = tuple(self.valid_loss_trace.x) + (self.epoch,)
            new_y = tuple(self.valid_loss_trace.y) + (loss,)
            self.valid_loss_trace.update(x=new_x, y=new_y)
            self.valid_num = 0
            self.valid_denom = 0
        
        range_changed = self.xy_range.update([self.epoch], [loss])
        if range_changed: self.update_range(self.xy_range.x_range(), self.xy_range.y_range())
        if not training: self.epoch += 1

    def after_fit(self):
        self.parent.register_user_epoch_event(self.train_loss_trace)
        self.parent.register_user_epoch_event(self.valid_loss_trace)
        
    def on_user_epoch(self, epoch:int):
        self.marker_trace.update(x=[self.parent.clicked_trace.x[epoch]], y=[self.parent.clicked_trace.y[epoch]])


class MetricSP(SubPlot):
    """Plots the specified metric as a function of epoch"""
    def __init__(self, metric_name:str, metric:Metric[Tensor], colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.metric_name = metric_name
        self.metric = metric
        self.xy_range = AxisRange(0, 10, 0, 0.001)
        self.metric_trace: BaseTraceType = None
        self.marker_trace: BaseTraceType = None
        self.epoch = 0
        self.train_num = 0
        self.train_denom = 0
        self.valid_num = 0
        self.valid_denom = 0
        
    def title(self) -> str: return self.metric_name
    def xlabel(self) -> str: return 'Epoch'
    def ylabel(self) -> str: return self.metric_name

    def create_empty(self, parent:PlotGrid, spi, position):
        super().create_empty(parent, spi, position)
        metric_trace = go.Scatter(x=[], y=[], mode='lines+markers', name=self.metric_name)
        marker_trace = go.Scatter(x=[], y=[], mode='markers', showlegend=False, hoverinfo='skip',
                                  marker=dict(color='rgba(0,0,0,0.2)', line=dict(color='black', width=2)))
        self.metric_trace = parent.add_trace(self, metric_trace)
        self.marker_trace = parent.add_trace(self, marker_trace)
    
    def after_batch(self, training, inputs, targets, predictions, loss):
        if training: return  # Only interested in validation metrics
        self.metric.update(predictions.detach().cpu(), targets.detach().cpu())

    def after_epoch(self, training):
        if training: return  # Only interested in validation metrics
        value = self.metric.compute()
        new_x = tuple(self.metric_trace.x) + (self.epoch,)
        new_y = tuple(self.metric_trace.y) + (value,)
        self.metric_trace.update(x=new_x, y=new_y)
        self.metric.reset()

        range_changed = self.xy_range.update([self.epoch], [value])
        if range_changed: self.update_range(self.xy_range.x_range(), self.xy_range.y_range())
        self.epoch += 1

    def after_fit(self):
        self.parent.register_user_epoch_event(self.metric_trace)
        self.parent.register_user_epoch_event(self.metric_trace)
        
    def on_user_epoch(self, epoch:int):
        y = self.metric_trace.y[epoch]
        self.marker_trace.update(x=[epoch], y=[y])

class ValidLossSP(SubPlot):
    """
    Scatter plot of validation loss for individual samples
    Unlike the standard loss functions `batch_loss_fn` must not perform
    a reduction (e.g. mean) over the samples in the batch at the end.

    For example: 
    batch_loss_fn = lambda preds,targs: F.nll_loss(preds, target, reduction=None)
    """
    def __init__(self, batch_loss_fn:Callable, colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.loss_fn = batch_loss_fn
        self.xy_range = AxisRange(0, 10, 0, 0.001)
        self.cur_epoch_loss: List[float] = []
        self.loss: List[List[float]] = []  # self.loss[epoch][sample]
        self.scatter_trace: BaseTraceType = None
        self.marker_trace: BaseTraceType = None
        self.user_epoch:int = None  # User-selected epoch
        self.user_sample:int = None  # User-selected sample index
        
    def title(self) -> str: return 'All-sample validation loss'
    def xlabel(self) -> str: return 'Sample'
    def ylabel(self) -> str: return 'Validation loss'

    def create_empty(self, parent:PlotGrid, spi, position):
        super().create_empty(parent, spi, position)
        scatter_trace = go.Scatter(x=[], y=[], mode='markers', showlegend=False)
        marker_trace = go.Scatter(x=[], y=[], mode='markers', showlegend=False, hoverinfo='skip',
                                  marker=dict(color='rgba(0,0,0,0.2)', line=dict(color='black', width=2)))
        self.scatter_trace = parent.add_trace(self, scatter_trace)
        self.marker_trace = parent.add_trace(self, marker_trace)
    
    def after_batch(self, training, inputs, targets, predictions, loss):
        if training: return  # Only interested in validation metrics
        loss:Tensor = self.loss_fn(predictions, targets)
        self.cur_epoch_loss += loss.tolist()

    def after_epoch(self, training):
        if training: return  # Only interested in validation metrics
        new_y = self.cur_epoch_loss.copy()
        new_x = list(range(len(new_y)))
        self.loss.append(self.cur_epoch_loss.copy())
        self.cur_epoch_loss = []  # Clear for next epoch

        self.scatter_trace.update(x=new_x, y=new_y)
        range_changed = self.xy_range.update(new_x, new_y)
        if range_changed: self.update_range(self.xy_range.x_range(), self.xy_range.y_range())

    def after_fit(self):
        self.parent.register_user_sample_event(self.scatter_trace)
        
    def on_user_sample(self, sample):
        self.user_sample = sample
        self.marker_trace.update(x=[self.parent.clicked_trace.x[sample]], y=[self.parent.clicked_trace.y[sample]])

    def on_user_epoch(self, epoch:int):
        self.user_epoch = epoch

        # Update scatter plot
        new_y = self.loss[epoch]
        new_x = list(range(len(new_y)))
        self.scatter_trace.update(x=new_x, y=new_y)

        # Update marker if applicable
        if self.user_sample is not None:
            self.marker_trace.update(x=[self.scatter_trace.x[self.user_sample]], y=[self.scatter_trace.y[self.user_sample]])

class ImageSP(SubPlot):
    """
    Visualize an image from a dataset
    """
    def __init__(self, ds:Dataset, sample_idx:int=0, colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.ds, self.sample_idx = ds, sample_idx
        self.sample_img:Tensor = self.ds[self.sample_idx][0]  # (C,H,W)
        self.img_trace: BaseTraceType = None
        
    def title(self) -> str: return f'Sample {self.sample_idx}'
    def xlabel(self) -> str: return ''
    def ylabel(self) -> str: return ''

    def create_empty(self, parent:PlotGrid, spi, position):
        # RGB images: https://plotly.com/python/imshow/#display-multichannel-image-data-with-goimage
        # Single-channel: https://plotly.com/python/heatmaps/#basic-heatmap-with-plotlygraphobjects
        # Color scales: https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales
        super().create_empty(parent, spi, position)
        C,H,W = self.sample_img.shape
        if C>1:
            z=self.sample_img.transpose((1,2,0)).tolist()  # Move channel dimension to end
            img_trace = go.Image(z=z)
        else:
            z=self.sample_img.tolist()[0]  # Remove channel dimension
            img_trace = go.Heatmap(z=z, showscale=False, colorscale='gray')
        self.img_trace = parent.add_trace(self, img_trace)
        self.update_range([0,W], [H,0])

        # Ensure square tiles
        xanchor_name = self.append_spi('x')
        yanchor_name = self.append_spi('y')
        self.update_axes(xaxis=dict(scaleanchor=yanchor_name, showgrid=False, zeroline=False),
                         yaxis=dict(scaleanchor=xanchor_name, showgrid=False, zeroline=False))
    
    def on_user_sample(self, sample:int):
        self.sample_idx = sample
        self.sample_img = self.ds[self.sample_idx][0]  # (C,H,W)

        C,H,W = self.sample_img.shape
        if C>1:
            z=self.sample_img.transpose((1,2,0)).tolist()  # Move channel dimension to end
            self.img_trace.update(z=z)
        else:
            z=self.sample_img.tolist()[0]  # Remove channel dimension
            self.img_trace.update(z=z)