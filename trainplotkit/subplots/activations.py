from typing import List, Tuple, Mapping, Callable, Any
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torcheval.metrics.metric import Metric
import plotly.graph_objects as go
import plotly.callbacks as cb
from plotly.basedatatypes import BaseTraceType
from ..plotgrid import PlotGrid, SubPlot
from ..utils import AxisRange, to_cpu

class ColorfulDimensionSP(SubPlot):
    """
    This is the colorful dimension activation histogram from the 2022 fastai
    course (https://course.fast.ai/Lessons/lesson16.html from 1:14:28). 
    It is similar to a histogram with a few changes:
    * A time (batch) dimension is added and becomes the new x-axis
    * The value axis (absolute value of activations of a particular module) 
      moves from the x-axis to the y-axis
    * The frequency of occurrences moves from the y-axis to the color axis

    Inputs
    * `module`: The module producing the activations to be analyzed.
       PyTorch hooks are registered in `before_fit` and removed in `after_fit`.
    * `append`: If True, append new epochs to the right. If False, replaces
       the previous epoch with the next. Old epochs can always be displayed via
       `on_user_epoch` if `remember_past_epochs` is True.
    * `remember_past_epochs`: If True, the subplot will remember the
      plot data for earlier epochs and allow user interaction via 
      `on_user_epoch`. It is up to the user to ensure that 
      `num_epochs * num_bins * num_batches` values will fit in memory.
      `remember_past_epochs` is ignored if `append` is True.
    * `num_bins`: The number of histogram bins to compute for each batch
    * `bin_range`: A (min,max) tuple specifying the range of histogram bins
    * `colorscale`: A string from the list at 
       https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales
       or any value that can be passed as-is for the parameter documented at
       https://plotly.com/python/reference/heatmap/#heatmap-colorscale

    After training has completed, the following user interactions are available:
    * If `remember_past_epochs` was set to `True`, a different epoch may be 
      selected by calling `on_user_epoch` or clicking any subplot in the same 
      PlotGrid that calls `PlotGrid.register_user_epoch_event`
    * Clicking this subplot calls `on_user_batch`, causing all subplots in
      the same `PlotGrid` that implements this method to be updated
    """
    def __init__(self, module:nn.Module, append:bool=False, remember_past_epochs:bool=False, 
                 num_bins:int=40, bin_range:Tuple[float, float] = (0,10), colorscale:Any='Viridis', 
                 colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.module, self.append, self.remember_past_epochs = module, append, remember_past_epochs
        self.num_bins, self.bin_range, self.colorscale = num_bins, bin_range, colorscale
        self.hook = None
        self.cur_epoch_hist: List[List[float]] = []
        self.hist_history: List[List[List[float]]] = []  # self.hist_history[epoch][batch][bin] OR self.hist_history[batch][bin]
        self.hist_trace: BaseTraceType = None
        self.marker_trace: BaseTraceType = None
        self.user_epoch:int = None  # User-selected epoch
        self.user_batch:int = None  # User-selected batch index

        # Store histogram bin centers
        center_offset = (self.bin_range[1]-self.bin_range[0]) / (self.num_bins*2)
        self.bin_centers = torch.linspace(self.bin_range[0]+center_offset, self.bin_range[1]-center_offset, self.num_bins)
        
    def title(self) -> str:  
        epoch_str = '' if self.user_epoch is None else f': epoch {self.user_epoch}'
        return f'Activation histogram per batch{epoch_str}'
    def xlabel(self) -> str: return 'Batch'
    def ylabel(self) -> str: return 'Bin'

    def create_empty(self, parent:PlotGrid, spi, position):
        hovertemplate=(
            'Batch: %{x}<br>' +
            'Bin: %{y}<br>' +
            'Bin center: %{customdata[1]}<br>' +
            'Count: %{customdata[0]}<br>' +
            '<extra></extra>'
        )
        super().create_empty(parent, spi, position)
        hist_trace = go.Heatmap(z=[], transpose=True, colorscale=self.colorscale, 
                                showscale=False,
                                hovertemplate=hovertemplate, customdata=[[]])
        marker_trace = go.Scatter(x=[], y=[], mode='markers', showlegend=False, hoverinfo='skip',
                                  marker=dict(color='rgba(0,0,0,0.2)', line=dict(color='black', width=2)))
        self.hist_trace = parent.add_trace(self, hist_trace)
        self.marker_trace = parent.add_trace(self, marker_trace)

    def before_fit(self):
        self.hook = self.module.register_forward_hook(self.hook_fcn)
        
    def hook_fcn(self, mod:nn.Module, inp, outp):
        if not mod.training: return  # Only interested in training batches
        acts:Tensor = to_cpu(outp)
        hist = acts.abs().histc(self.num_bins, self.bin_range[0], self.bin_range[1]).float().log1p()
        self.cur_epoch_hist.append(hist.tolist())
        
    def after_epoch(self, training):
        if not training: return  # Only interested in training batches

        # Update state
        if self.append:
            self.hist_history += self.cur_epoch_hist.copy()
            new_z = self.hist_history
        else:
            if not self.remember_past_epochs: self.hist_history = []
            self.hist_history.append(self.cur_epoch_hist.copy())
            new_z = self.cur_epoch_hist.copy()
        self.cur_epoch_hist = []  # Clear for next epoch

        # Determine actual counts (reverse log1p)
        custom_count = (Tensor(new_z).exp() - 1).round().t()  # Transpose to [bin,batch]

        # Stack bin centers onto custom_count along new dimension
        num_batches = len(new_z)
        bin_centers_expanded = self.bin_centers[:,None].expand(-1, num_batches)
        new_custom_data = torch.stack((custom_count, bin_centers_expanded), dim=-1).tolist()

        # Update trace
        self.hist_trace.update(z=new_z, customdata=new_custom_data)

    def after_fit(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
        # TODO: get `register_user_batch_event` working for heatmap traces
        # self.parent.register_user_batch_event(self.hist_trace)
        
    def on_user_batch(self, batch):
        self.user_batch = batch
        self.marker_trace.update(x=[batch], y=[2.])

    def on_user_epoch(self, epoch:int):
        if self.append: return
        if not self.remember_past_epochs: return
        self.user_epoch = epoch

        # Update heatmap
        new_z = self.hist_history[epoch]
        self.hist_trace.update(z=new_z)

        # Update title
        self.update_title()

class ActivationStatsSP(SubPlot):
    """
    This subplot displays the means, standard deviations or dead charts of
    specified modules (layers) as a function of batch number as shown in the 2022
    fastai course (https://course.fast.ai/Lessons/lesson16.html) from 1:14:28. 

    The colorful dimension histogram shown in the same course is implemented
    via ColorfulDimensionSP.

    Inputs
    * `modules`: A list of modules for which the activation stats should be displayed.
       PyTorch hooks are registered in `before_fit` and removed in `after_fit`.
    * `statistic`: A string describing the statistic to measure. Valid options include:
      * `'mean'`: Display the mean activations of the specified modules
      * `'std'`: Display the standard deviations of the specified layers
      * `'dead'`: Display the dead charts of the specified layers as a fraction of 
        activations below the specified cutoff value.
      * `'exploding'`: Display the fraction of activations above the specified cutoff value.
    * `cutoff`: Cutoff value to use if `statistic` is either `dead` or `exploding`.
      For `'dead'`, a default value of 0.5 is used. This is equivalent to the fraction of 
      activations in the first two bins among 40 bins spaced between 0 and 10, which 
      is the parameters used in the fastai course. For `'exploding'`, a default value of
      2.0 is used.
    * `append`: If True, append new epochs to the right. If False, replaces
       the previous epoch with the next. Old epochs can always be displayed via
       `on_user_epoch` if `remember_past_epochs` is True.
    * `remember_past_epochs`: If True, the subplot will remember the
      plot data for earlier epochs and allow user interaction via 
      `on_user_epoch`. `remember_past_epochs` is ignored if `append` is True.

    After training has completed, the following user interactions are available:
    * If `remember_past_epochs` was set to `True`, a different epoch may be 
      selected by calling `on_user_epoch` or clicking any subplot in the same 
      PlotGrid that calls `PlotGrid.register_user_epoch_event`
    * Clicking this subplot calls `on_user_batch`, causing all subplots in
      the same `PlotGrid` that implements this method to be updated
    """
    def __init__(self, modules:List[nn.Module], statistic:str='std', cutoff:float=None,
                 append:bool=False, remember_past_epochs:bool=False, 
                 colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.modules, self.statistic, self.cutoff = modules, statistic, cutoff
        self.append, self.remember_past_epochs = append, remember_past_epochs
        if self.cutoff is None and self.statistic == 'dead': self.cutoff = 0.5
        if self.cutoff is None and self.statistic == 'exploding': self.cutoff = 2.0
        self.hooks:List = []
        self.cur_epoch_stats: List[List[float]] = [[] for _ in self.modules]  # self.cur_epoch_stats[layer][batch]
        self.stats_history: List[List[List[float]]] = []  # self.stats_history[epoch][layer][batch] OR self.stats_history[layer][batch]
        self.stats_traces: List[BaseTraceType] = []
        self.marker_trace: BaseTraceType = None
        self.user_epoch:int = None  # User-selected epoch
        self.user_batch:int = None  # User-selected batch index
        if self.append: self.stats_history = [[] for _ in self.modules] # self.stats_history[layer][batch]

    def title(self) -> str:  
        epoch_str = '' if self.user_epoch is None else f': epoch {self.user_epoch}'
        stat_str = 'dead chart' if self.statistic == 'dead' else \
                   'exploding chart' if self.statistic == 'exploding' else \
                   'standard deviation' if self.statistic == 'std' else self.statistic
        return f'Activation {stat_str}{epoch_str}'
    def xlabel(self) -> str: return 'Batch'
    def ylabel(self) -> str: return self.statistic

    def create_empty(self, parent:PlotGrid, spi, position):
        super().create_empty(parent, spi, position)
        for si,mod in enumerate(self.modules):
            stats_trace = go.Scatter(x=[], y=[], mode='lines', name=str(si))
            stats_trace = parent.add_trace(self, stats_trace)
            self.stats_traces.append(stats_trace)
            
        marker_trace = go.Scatter(x=[], y=[], mode='markers', showlegend=False, hoverinfo='skip',
                                  marker=dict(color='rgba(0,0,0,0.2)', line=dict(color='black', width=2)))
        self.marker_trace = parent.add_trace(self, marker_trace)

    def before_fit(self):
        for mod in self.modules:
            hook = mod.register_forward_hook(self.hook_fcn)
            self.hooks.append(hook)
        
    def hook_fcn(self, mod:nn.Module, inp, outp):
        if not mod.training: return  # Only interested in training batches
        if mod not in self.modules: return  # Prevents an exception later
        acts:Tensor = to_cpu(outp)
        if self.statistic == 'mean':
            value = acts.mean()
        elif self.statistic == 'std':
            value = acts.std()
        elif self.statistic == 'dead':
            value = (acts < self.cutoff).float().mean()
        elif self.statistic == 'exploding':
            value = (acts > self.cutoff).float().mean()
        else:
            raise ValueError(f'Unknown statistic: {self.statistic}')
        mod_idx = self.modules.index(mod)
        self.cur_epoch_stats[mod_idx].append(value)
        
    def after_epoch(self, training):
        if not training: return  # Only interested in training batches

        # Update state
        if self.append:
            for mod_idx in range(len(self.modules)):
                self.stats_history[mod_idx] += self.cur_epoch_stats[mod_idx].copy()
            new_yy = self.stats_history
        else:
            if not self.remember_past_epochs: self.stats_history = []
            self.stats_history.append(self.cur_epoch_stats.copy())
            new_yy = self.cur_epoch_stats.copy()
        self.cur_epoch_stats = [[] for _ in self.modules]  # Clear for next epoch

        # Update traces
        for mod_idx, stats_trace in enumerate(self.stats_traces):
            new_y = new_yy[mod_idx]
            new_x = list(range(len(new_y)))
            stats_trace.update(x=new_x, y=new_y)

    def after_fit(self):
        for hook in self.hooks: hook.remove()
        self.hooks = []
        for stats_trace in self.stats_traces:
            self.parent.register_user_batch_event(stats_trace)
        
    def on_user_batch(self, batch):
        self.user_batch = batch
        if self.parent.clicked_trace in self.stats_traces:
            y = self.parent.clicked_trace.y[batch]
        else:
            y = self.stats_traces[0].y[batch]
        self.marker_trace.update(x=[batch], y=[y])

    def on_user_epoch(self, epoch:int):
        if self.append: return
        if not self.remember_past_epochs: return
        self.user_epoch = epoch

        # Update scatter plots
        new_yy = self.stats_history[epoch]
        for mod_idx, stats_trace in enumerate(self.stats_traces):
            new_y = new_yy[mod_idx]
            new_x = list(range(len(new_y)))
            stats_trace.update(x=new_x, y=new_y)

        # Update title
        self.update_title()



