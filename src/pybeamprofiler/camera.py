"""Base camera interface for beam profiler."""

from abc import ABC, abstractmethod

import ipywidgets as widgets
import numpy as np
from IPython.display import display


class Camera(ABC):
    """Abstract base class for camera interfaces.

    Defines the common interface for all camera types (simulated,
    FLIR, Basler, etc.).
    """

    def __init__(self):
        self.exposure_time = 0.01  # seconds
        self.gain = 0.0
        self.is_acquiring = False
        self.width = 0
        self.height = 0
        self.pixel_size = 1.0  # um
        self.image_buffer = None

    @abstractmethod
    def open(self):
        """Open connection to the camera."""
        pass

    @abstractmethod
    def close(self):
        """Close connection to the camera."""
        pass

    @abstractmethod
    def start_acquisition(self):
        """Start image acquisition."""
        pass

    @abstractmethod
    def stop_acquisition(self):
        """Stop image acquisition."""
        pass

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """Get a single image from the camera."""
        pass

    @abstractmethod
    def set_exposure(self, exposure_time: float):
        """Set exposure time in seconds."""
        pass

    @abstractmethod
    def set_gain(self, gain: float):
        """Set gain."""
        pass

    def setting(self):
        """Display interactive camera controls in Jupyter Notebook.

        Creates tabbed interface with exposure, gain, and acquisition controls.
        """
        style = {"description_width": "initial"}

        # Get exposure range from camera if available
        exposure_min, exposure_max = 1e-6, 1.0  # Default: 1us to 1s
        if hasattr(self, "exposure_range"):
            exposure_min, exposure_max = self.exposure_range

        gain_min, gain_max = 0.0, 24.0
        if hasattr(self, "gain_range"):
            gain_min, gain_max = self.gain_range

        import math

        exp_min_log = math.floor(math.log10(exposure_min))
        exp_max_log = math.ceil(math.log10(exposure_max))

        exposure_slider = widgets.FloatLogSlider(
            value=self.exposure_time,
            base=10,
            min=exp_min_log,
            max=exp_max_log,
            step=0.1,
            description="Exposure (s):",
            style=style,
            readout_format=".6f",
        )

        exposure_input = widgets.FloatText(
            value=self.exposure_time,
            description="",
            step=0.001,
            layout=widgets.Layout(width="100px"),
        )

        # Gain settings
        gain_slider = widgets.FloatSlider(
            value=self.gain,
            min=gain_min,
            max=gain_max,
            step=0.1,
            description="Gain (dB):",
            style=style,
        )

        gain_input = widgets.FloatText(
            value=self.gain,
            description="",
            step=0.1,
            layout=widgets.Layout(width="100px"),
        )

        start_button = widgets.Button(
            description="Start Acquisition", button_style="success", icon="play"
        )

        stop_button = widgets.Button(
            description="Stop Acquisition",
            button_style="danger",
            icon="stop",
            disabled=True,
        )

        def on_exposure_change(change):
            self.set_exposure(change["new"])
            exposure_input.value = change["new"]

        def on_gain_change(change):
            self.set_gain(change["new"])
            gain_input.value = change["new"]

        def on_exposure_input_change(change):
            exposure_slider.value = change["new"]

        def on_gain_input_change(change):
            gain_slider.value = change["new"]

        def on_start_click(b):
            self.start_acquisition()
            start_button.disabled = True
            stop_button.disabled = False

        def on_stop_click(b):
            self.stop_acquisition()
            start_button.disabled = False
            stop_button.disabled = True

        exposure_slider.observe(on_exposure_change, names="value")
        gain_slider.observe(on_gain_change, names="value")
        exposure_input.observe(on_exposure_input_change, names="value")
        gain_input.observe(on_gain_input_change, names="value")
        start_button.on_click(on_start_click)
        stop_button.on_click(on_stop_click)

        exposure_box = widgets.HBox([exposure_slider, exposure_input])
        gain_box = widgets.HBox([gain_slider, gain_input])

        timing_accordion = widgets.Accordion(children=[exposure_box])
        timing_accordion.set_title(0, "Exposure Time")

        analog_accordion = widgets.Accordion(children=[gain_box])
        analog_accordion.set_title(0, "Gain")

        acquisition_accordion = widgets.Accordion(
            children=[widgets.HBox([start_button, stop_button])]
        )
        acquisition_accordion.set_title(0, "Acquisition Control")

        tab = widgets.Tab()
        tab.children = [
            widgets.VBox([timing_accordion, analog_accordion]),
            acquisition_accordion,
        ]
        tab.set_title(0, "Camera Settings")
        tab.set_title(1, "Acquisition")

        display(tab)
