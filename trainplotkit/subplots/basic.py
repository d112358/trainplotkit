from typing import List, Tuple, Mapping, Callable, Any
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torcheval.metrics.metric import Metric
import numpy as np
import plotly.graph_objects as go
import plotly.callbacks as cb
from plotly.basedatatypes import BaseTraceType
from ..plotgrid import PlotGrid, SubPlot
from ..utils import AxisRange

class TrainingCurveSP(SubPlot):
    """
    Plots the training and validation loss as a function of epoch

    After training has completed, the following user interactions are available:
    * Clicking this subplot calls `on_user_epoch`, causing all subplots in
      the same `PlotGrid` that implements this method to be updated
    """
    def __init__(self, colspan:int=1, rowspan:int=1):
        super().__init__(colspan, rowspan)
        self.xy_range = AxisRange(0, 10, 0, 0.001)
        self.train_loss_trace: BaseTraceType = None
        self.valid_loss_trace: BaseTraceType = None
        self.marker_trace: BaseTraceType = None
        self.train_epoch = 0
        self.train_num = 0
        self.train_denom = 0
        self.valid_epoch = 0
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
            new_x = tuple(self.train_loss_trace.x) + (self.train_epoch,)
            new_y = tuple(self.train_loss_trace.y) + (loss,)
            self.train_loss_trace.update(x=new_x, y=new_y)
            self.train_num = 0
            self.train_denom = 0
            range_changed = self.xy_range.update([self.train_epoch], [loss])
            if range_changed: self.update_range(self.xy_range.x_range(), self.xy_range.y_range())
            self.train_epoch += 1
        else:
            loss = self.valid_num / self.valid_denom
            new_x = tuple(self.valid_loss_trace.x) + (self.valid_epoch,)
            new_y = tuple(self.valid_loss_trace.y) + (loss,)
            self.valid_loss_trace.update(x=new_x, y=new_y)
            self.valid_num = 0
            self.valid_denom = 0
            range_changed = self.xy_range.update([self.valid_epoch], [loss])
            if range_changed: self.update_range(self.xy_range.x_range(), self.xy_range.y_range())
            self.valid_epoch += 1

    def after_fit(self):
        self.parent.register_user_epoch_event(self.train_loss_trace)
        self.parent.register_user_epoch_event(self.valid_loss_trace)
        
    def on_user_epoch(self, epoch:int):
        self.marker_trace.update(x=[self.parent.clicked_trace.x[epoch]], y=[self.parent.clicked_trace.y[epoch]])


class MetricSP(SubPlot):
    """
    Plots the specified metric as a function of epoch

    After training has completed, the following user interactions are available:
    * Clicking this subplot calls `on_user_epoch`, causing all subplots in
      the same `PlotGrid` that implements this method to be updated
    """
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
        if training: return  # Only interested in validation samples
        self.metric.update(predictions.detach().cpu(), targets.detach().cpu())

    def after_epoch(self, training):
        if training: return  # Only interested in validation samples
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
    Scatter plot of validation loss for individual samples as a function of 
    sample index

    Usage notes:
    * This subplot considers only the validation set and its interpretation 
      may be counter-intuitive if the validation set is shuffled between epochs

    Inputs
    * `batch_loss_fn`: A callable of the following form:
      `loss = batch_loss_fn(predictions, targets)`
      where `preditions`, `targets` and `loss` are all tensors representing a 
      single batch of data. Unlike the standard loss functions provided by 
      PyTorch, `batch_loss_fn` must not perform a reduction (e.g. mean) along the 
      batch dimension. An example of a suitable `batch_loss_fn` is given as:
      `batch_loss_fn = lambda preds,targs: F.nll_loss(preds, target, reduction=None)`
    * `remember_past_epochs`: If True, the subplot will remember the
      validation loss for earlier epochs and allow user interaction via 
      `on_user_epoch`. It is up to the user to ensure that 
      `num_epochs * num_samples` values will fit in memory.

    After training has completed, the following user interactions are available:
    * If `remember_past_epochs` was set to `True`, a different epoch may be 
      selected by calling `on_user_epoch` or clicking any subplot in the same 
      PlotGrid that calls `PlotGrid.register_user_epoch_event`
    * Clicking this subplot calls `on_user_sample`, causing all subplots in
      the same `PlotGrid` that implements this method to be updated
    """
    def __init__(self, batch_loss_fn:Callable, remember_past_epochs:bool, colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.loss_fn = batch_loss_fn
        self.remember_past_epochs = remember_past_epochs
        self.xy_range = AxisRange(0, 10, 0, 0.001)
        self.cur_epoch_loss: List[float] = []
        self.loss: List[List[float]] = []  # self.loss[epoch][sample]
        self.scatter_trace: BaseTraceType = None
        self.marker_trace: BaseTraceType = None
        self.user_epoch:int = None  # User-selected epoch
        self.user_sample:int = None  # User-selected sample index
        
    def title(self) -> str:  
        epoch_str = '' if self.user_epoch is None else f': epoch {self.user_epoch}'
        return f'Validation loss per sample{epoch_str}'
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
        if training: return  # Only interested in validation samples
        loss:Tensor = self.loss_fn(predictions.detach().cpu(), targets.detach().cpu())
        self.cur_epoch_loss += loss.tolist()

    def after_epoch(self, training):
        if training: return  # Only interested in validation samples
        new_y = self.cur_epoch_loss.copy()
        new_x = list(range(len(new_y)))
        if not self.remember_past_epochs: self.loss = []
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
        if not self.remember_past_epochs: return
        self.user_epoch = epoch

        # Update scatter plot
        new_y = self.loss[epoch]
        new_x = list(range(len(new_y)))
        self.scatter_trace.update(x=new_x, y=new_y)

        # Update marker if applicable
        if self.user_sample is not None:
            self.marker_trace.update(x=[self.scatter_trace.x[self.user_sample]], y=[self.scatter_trace.y[self.user_sample]])

        # Update title
        self.update_title()


class ImageSP(SubPlot):
    """
    Visualize an image from a dataset

    Usage notes:
    * This subplot considers only the validation set and its interpretation 
      may be counter-intuitive if the validation set is shuffled between epochs

    After training has completed, the following user interactions are available:
    * A different sample may be selected by calling `on_user_sample` or clicking
      any subplot in the same PlotGrid that calls 
      `PlotGrid.register_user_sample_event`
    """
    def __init__(self, ds:Dataset, sample_idx:int=0, class_names:List[str]=None, colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.ds, self.sample_idx = ds, sample_idx
        self.sample_img:Tensor = self.get_image(self.sample_idx)  # (C,H,W)
        self.class_names = class_names
        self.img_trace: BaseTraceType = None
        
    def title(self) -> str:
        target = self.get_target(self.sample_idx)
        if self.class_names: target = self.class_names[target]
        return f'Input: sample {self.sample_idx}<br>Target={target}'
    def xlabel(self) -> str: return ''
    def ylabel(self) -> str: return ''

    def get_image(self, sample_idx):
        # Extract from list, tuple or dictionary
        sample = self.ds[sample_idx]
        if isinstance(sample, Mapping):
            img_raw = sample['image']
        else:
            img_raw = sample[0]

        # Convert to Tensor if not already
        if isinstance(img_raw, Tensor): 
            img = img_raw
        elif isinstance(img_raw, np.ndarray): 
            img = Tensor(img_raw)
        else:
            img = Tensor(np.array(img_raw))

        # Ensure shape is (C,H,W)
        if img.ndim == 2:
            return img[None,:,:]
        elif img.ndim == 3:
            if img.shape[0] in [1,3,4]:
                return img
            elif img.shape[2] in [1,3,4]: 
                return img.permute((2,0,1))
            else:
                raise ValueError(f'Unrecognized shape for image tensor: {img.shape}')
        else:
            raise ValueError(f'Unrecognized shape for image tensor: {img.shape}')

    def get_target(self, sample_idx):
        sample = self.ds[sample_idx]
        if isinstance(sample, Mapping):
            return sample['label']
        else:
            return sample[1]

    def create_empty(self, parent:PlotGrid, spi, position):
        # RGB images: https://plotly.com/python/imshow/#display-multichannel-image-data-with-goimage
        # Single-channel: https://plotly.com/python/heatmaps/#basic-heatmap-with-plotlygraphobjects
        # Color scales: https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales
        super().create_empty(parent, spi, position)
        C,H,W = self.sample_img.shape
        z = self.sample_img.permute((1,2,0))  # Move channel dimension to end
        if C==1: z = z.tile((1,1,3))          # Repeat channel dimension for single-channel images
        zmin = [float(self.sample_img.min())] * 3 + [0]
        zmax = [float(self.sample_img.max())] * 3 + [1]
        img_trace = go.Image(z=z.tolist(), zmin=zmin, zmax=zmax)
        self.img_trace = parent.add_trace(self, img_trace)

        # Ensure square tiles
        # * At the moment, plotly seems to handle all of this when an image trace is added to a
        #   subplot: https://plotly.com/python/reference/image/
        square_manually = False
        if square_manually:
            self.update_range([0,W], [H,0])
            xanchor_name = self.append_spi('x')
            self.update_axes(xaxis=dict(showgrid=False, zeroline=False),
                             yaxis=dict(scaleanchor=xanchor_name, showgrid=False, zeroline=False))
        
    def before_show_static(self):
        """
        Sometimes a static figure created via `PlotGrid.show_static()` shows images upside down if
        generated after the user has interacted with the Plotly controls on the figure, e.g.
        auto-scale. 
        
        Just assigning the y-range to the same value again updates internal state variables to
        show it upright again.
        """
        yaxis_name = self.append_spi('yaxis')
        self.parent.widget.layout[yaxis_name]['range'] = self.parent.widget.layout[yaxis_name]['range']

    def on_user_sample(self, sample:int):
        self.sample_idx = sample
        self.sample_img = self.get_image(self.sample_idx)  # (C,H,W)

        C,H,W = self.sample_img.shape
        z = self.sample_img.permute((1,2,0))  # Move channel dimension to end
        if C==1: z = z.tile((1,1,3))          # Repeat channel dimension for single-channel images
        zmin = [float(self.sample_img.min())] * 3 + [0]
        zmax = [float(self.sample_img.max())] * 3 + [1]
        self.img_trace.update(z=z.tolist(), zmin=zmin, zmax=zmax)
        self.update_title()

class PredImageSP(SubPlot):
    """
    Visualize a predicted image

    Usage notes:
    * This subplot considers only the validation set and its interpretation 
      may be counter-intuitive if the validation set is shuffled between epochs

    Inputs
    * `remember_past_epochs`: If True, the subplot will remember class
      probabilities for earlier epochs and allow user interaction via 
      `on_user_epoch`. It is up to the user to ensure that 
      `num_epochs * N*C*H*W` values will fit in memory.
    * `img_size`: A (height,width) tuple used to reshape the image if these two
       dimensions were combined in the output
    * `sample_idx`: The index of the validation sample to display initially. This 
      can be modified interactively after training has completed 

    After training has completed, the following user interactions are available:
    * A different sample may be selected by calling `on_user_sample` or clicking
      any subplot in the same PlotGrid that calls 
      `PlotGrid.register_user_sample_event`
    * If `remember_past_epochs` was set to `True`, a different epoch may be 
      selected by calling `on_user_epoch` or clicking any subplot in the same 
      PlotGrid that calls `PlotGrid.register_user_epoch_event`
    """
    def __init__(self, remember_past_epochs:bool, img_size:Tuple[int]=None, sample_idx:int=0, colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.remember_past_epochs, self.img_size = remember_past_epochs, img_size
        self.cur_epoch_images: Tensor = Tensor([])  # self.cur_epoch_images[sample_idx,channel_idx,y,x]
        self.images: List[Tensor] = []  # self.images[epoch][sample_idx,channel_idx,y,x]
        self.img_trace: BaseTraceType = None
        self.user_epoch:int = None  # User-selected epoch
        self.user_sample:int = sample_idx  # User-selected sample index
        
    def title(self) -> str:
        epoch_str = '' if self.user_epoch is None else f'epoch {self.user_epoch}, '
        sample_str = 'sample 0' if self.user_sample is None else f'sample {self.user_sample}'
        return f'Predicted: {epoch_str}{sample_str}'
    def xlabel(self) -> str: return ''
    def ylabel(self) -> str: return ''

    def create_empty(self, parent:PlotGrid, spi, position):
        # RGB images: https://plotly.com/python/imshow/#display-multichannel-image-data-with-goimage
        # Single-channel: https://plotly.com/python/heatmaps/#basic-heatmap-with-plotlygraphobjects
        # Color scales: https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales
        super().create_empty(parent, spi, position)
        img_trace = go.Image(z=[])
        self.img_trace = parent.add_trace(self, img_trace)

    def preproc_img(self, img:Tensor):
        C,H,W = img.shape
        z = img.permute((1,2,0))  # Move channel dimension to end
        if C==1: z = z.tile((1,1,3))          # Repeat channel dimension for single-channel images
        zmin = [float(img.min())] * 3 + [0]
        zmax = [float(img.max())] * 3 + [1]
        return z.tolist(), zmin, zmax
        
    def after_batch(self, training, inputs, targets, predictions:Tensor, loss):
        if training: return  # Only interested in validation samples
        must_reshape = self.img_size is not None and self.img_size[-1] != predictions.shape[-1]
        new_shape = predictions.shape[:-1] + self.img_size if must_reshape else predictions.shape
        preds_hw = predictions.view(new_shape)
        preds_nchw = preds_hw.unsqueeze(dim=1) if preds_hw.ndim < 4 else preds_hw  # [N,C,H,W]
        self.cur_epoch_images = torch.cat((self.cur_epoch_images, preds_nchw), dim=0)

    def after_epoch(self, training):
        if training: return  # Only interested in validation samples
        img:Tensor = self.cur_epoch_images[self.user_sample].clone() # [C,H,W]
        if not self.remember_past_epochs: self.images = []
        self.images.append(self.cur_epoch_images.clone())
        self.cur_epoch_images = Tensor([])  # Clear for next epoch
        
        z, zmin, zmax = self.preproc_img(img)
        self.img_trace.update(z=z, zmin=zmin, zmax=zmax)

    def before_show_static(self):
        # See ImageSP.before_show_static
        yaxis_name = self.append_spi('yaxis')
        self.parent.widget.layout[yaxis_name]['range'] = self.parent.widget.layout[yaxis_name]['range']

    def on_user_sample(self, sample):
        epoch = self.user_epoch if self.user_epoch is not None else -1
        self.user_sample = sample

        # Update image and title
        img:Tensor = self.images[epoch][sample]
        z, zmin, zmax = self.preproc_img(img)
        self.img_trace.update(z=z, zmin=zmin, zmax=zmax)
        self.update_title()

    def on_user_epoch(self, epoch:int):
        if not self.remember_past_epochs: return
        self.user_epoch = epoch
        sample = self.user_sample if self.user_sample is not None else 0

        # Update image and title
        img:Tensor = self.images[epoch][sample]
        z, zmin, zmax = self.preproc_img(img)
        self.img_trace.update(z=z, zmin=zmin, zmax=zmax)
        self.update_title()


class ClassProbsSP(SubPlot):
    """
    Bar graph of class probabilities in a classification task

    Usage notes:
    * This subplot considers only the validation set and its interpretation 
      may be counter-intuitive if the validation set is shuffled between epochs

    Inputs
    * `probs_fn`: A callable of the following form:
      `probs = probs_fn(predictions)`
      where `predictions` is the parameter sent to `after_batch` and
      `probs` is a tensor of size `(num_samples, num_classes)` specifying
      the probabilities of all classes for all samples in the current batch.
      - If the model predictions are logits, the following is recommended:
        `prob_fn = lambda preds: torch.softmax(preds, dim=1)`
      - If the model predictions are already probabilities, `None` may be
        specified here.
    * `remember_past_epochs`: If True, the subplot will remember class
      probabilities for earlier epochs and allow user interaction via 
      `on_user_epoch`. It is up to the user to ensure that 
      `num_epochs * num_samples * num_classes` values will fit in memory.
    * `class_names`: Names to display on the x-axis. If omitted, class 
      indices will be displayed

    After training has completed, the following user interactions are available:
    * A different sample may be selected by calling `on_user_sample` or clicking
      any subplot in the same PlotGrid that calls 
      `PlotGrid.register_user_sample_event`
    * If `remember_past_epochs` was set to `True`, a different epoch may be 
      selected by calling `on_user_epoch` or clicking any subplot in the same 
      PlotGrid that calls `PlotGrid.register_user_epoch_event`
    """
    def __init__(self, probs_fn:Callable, remember_past_epochs:bool, class_names:List[str]=None, colspan=1, rowspan=1):
        super().__init__(colspan, rowspan)
        self.probs_fn = probs_fn
        self.remember_past_epochs = remember_past_epochs
        self.class_names = class_names
        self.cur_epoch_probs: List[List[float]] = []  # self.cur_epoch_probs[sample][class]
        self.probs: List[List[List[float]]] = []  # self.loss[epoch][sample][class]
        self.bar_trace: BaseTraceType = None
        self.user_epoch:int = None  # User-selected epoch
        self.user_sample:int = None  # User-selected sample index
        
    def title(self) -> str:
        epoch_str = '' if self.user_epoch is None else f'epoch {self.user_epoch}, '
        sample_str = 'sample 0' if self.user_sample is None else f'sample {self.user_sample}'
        return f'Class probabilities: {epoch_str}{sample_str}'
    def xlabel(self) -> str: return 'Class'
    def ylabel(self) -> str: return 'Probability'

    def create_empty(self, parent:PlotGrid, spi, position):
        super().create_empty(parent, spi, position)
        bar_trace = go.Bar(x=[], y=[], showlegend=False)
        self.bar_trace = parent.add_trace(self, bar_trace)
    
    def after_batch(self, training, inputs, targets, predictions:Tensor, loss):
        if training: return  # Only interested in validation samples
        probs:Tensor = self.probs_fn(predictions.detach().cpu()) if self.probs_fn else predictions.detach().cpu().flatten(start_dim=1)
        self.cur_epoch_probs += probs.tolist()

    def after_epoch(self, training):
        if training: return  # Only interested in validation samples
        new_y = self.cur_epoch_probs[0].copy()
        new_x = self.class_names if self.class_names else list(range(len(new_y)))
        if not self.remember_past_epochs: self.probs = []
        self.probs.append(self.cur_epoch_probs.copy())
        self.cur_epoch_probs = []  # Clear for next epoch
        self.bar_trace.update(x=new_x, y=new_y)
        
    def on_user_sample(self, sample):
        epoch = self.user_epoch if self.user_epoch is not None else -1
        self.user_sample = sample

        # Update bar graph and title
        new_y = self.probs[epoch][sample]
        new_x = self.class_names if self.class_names else list(range(len(new_y)))
        self.bar_trace.update(x=new_x, y=new_y)
        self.update_title()

    def on_user_epoch(self, epoch:int):
        if not self.remember_past_epochs: return
        self.user_epoch = epoch
        sample = self.user_sample if self.user_sample is not None else 0

        # Update bar graph and title
        new_y = self.probs[epoch][sample]
        new_x = self.class_names if self.class_names else list(range(len(new_y)))
        self.bar_trace.update(x=new_x, y=new_y)
        self.update_title()

