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

    def setting(self, **kwargs):
        """Display interactive camera controls in Jupyter Notebook.

        Creates tabbed interface with exposure, gain, and acquisition controls.
        Dynamically populates controls from GenICam node_map for real cameras.

        Args:
            **kwargs: Optional keyword arguments to set camera parameters.
                     Parameter names should match node_map feature names.
                     Examples: ExposureTime=0.01, Gain=10.0, BlackLevel=0
        """
        # Apply settings from kwargs if provided
        if kwargs:
            self._apply_settings_from_kwargs(kwargs)

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

        # Additional GenICam controls (populated dynamically from node_map)
        genicam_controls = self._create_genicam_controls(style)
        advanced_controls = self._create_advanced_controls(style)

        # Camera information display
        camera_info = []
        camera_info.append(widgets.HTML(f"<b>Camera Type:</b> {type(self).__name__}"))
        camera_info.append(widgets.HTML(f"<b>Sensor Size:</b> {self.width}×{self.height} pixels"))
        camera_info.append(widgets.HTML(f"<b>Pixel Size:</b> {self.pixel_size:.2f} μm"))

        # Add sensor description for GenICam cameras
        if hasattr(self, "node_map") and self.node_map:
            try:
                if hasattr(self.node_map, "SensorDescription"):
                    desc = self.node_map.SensorDescription.value
                    camera_info.append(widgets.HTML(f"<b>Sensor:</b> {desc}"))
            except Exception:
                pass
            try:
                if hasattr(self.node_map, "DeviceModelName"):
                    model = self.node_map.DeviceModelName.value
                    camera_info.append(widgets.HTML(f"<b>Model:</b> {model}"))
            except Exception:
                pass

        camera_info_box = widgets.VBox(camera_info)

        # ROI controls (for GenICam cameras)
        roi_controls = None
        if hasattr(self, "roi_info") and hasattr(self, "set_roi"):
            roi = self.roi_info

            offset_x_input = widgets.IntText(
                value=roi["offset_x"], description="Offset X:", style=style
            )
            offset_y_input = widgets.IntText(
                value=roi["offset_y"], description="Offset Y:", style=style
            )
            width_input = widgets.IntText(value=roi["width"], description="Width:", style=style)
            height_input = widgets.IntText(value=roi["height"], description="Height:", style=style)

            roi_button = widgets.Button(
                description="Apply ROI", button_style="primary", icon="check"
            )

            roi_reset_button = widgets.Button(
                description="Full Sensor", button_style="info", icon="arrows-alt"
            )

            def on_roi_apply(b):
                try:
                    self.set_roi(
                        offset_x_input.value,
                        offset_y_input.value,
                        width_input.value,
                        height_input.value,
                    )
                    # Update info display
                    updated_roi = self.roi_info
                    camera_info[
                        1
                    ].value = (
                        f"<b>Sensor Size:</b> {updated_roi['width']}×{updated_roi['height']} pixels"
                    )
                except Exception as e:
                    print(f"Error setting ROI: {e}")

            def on_roi_reset(b):
                try:
                    roi_max = self.roi_info
                    self.set_roi(0, 0, roi_max["max_width"], roi_max["max_height"])
                    offset_x_input.value = 0
                    offset_y_input.value = 0
                    width_input.value = roi_max["max_width"]
                    height_input.value = roi_max["max_height"]
                    camera_info[
                        1
                    ].value = (
                        f"<b>Sensor Size:</b> {roi_max['max_width']}×{roi_max['max_height']} pixels"
                    )
                except Exception as e:
                    print(f"Error resetting ROI: {e}")

            roi_button.on_click(on_roi_apply)
            roi_reset_button.on_click(on_roi_reset)

            roi_controls = widgets.VBox(
                [
                    widgets.HTML(f"<b>ROI Max:</b> {roi['max_width']}×{roi['max_height']} pixels"),
                    offset_x_input,
                    offset_y_input,
                    width_input,
                    height_input,
                    widgets.HBox([roi_button, roi_reset_button]),
                ]
            )

        # Build tab layout
        settings_children = [timing_accordion, analog_accordion]

        # Add GenICam controls if available
        if genicam_controls:
            settings_children.extend(genicam_controls)

        if roi_controls:
            roi_accordion = widgets.Accordion(children=[roi_controls])
            roi_accordion.set_title(0, "Region of Interest")
            settings_children.append(roi_accordion)

        info_accordion = widgets.Accordion(children=[camera_info_box])
        info_accordion.set_title(0, "Camera Information")

        # Create tabs
        tab = widgets.Tab()
        tab_children = [
            widgets.VBox(settings_children),
            info_accordion,
        ]

        # Add advanced tab if there are advanced controls
        if advanced_controls:
            advanced_tab = widgets.VBox(advanced_controls)
            tab_children.append(advanced_tab)

        tab.children = tab_children
        tab.set_title(0, "Camera Settings")
        tab.set_title(1, "Info")
        if advanced_controls:
            tab.set_title(2, "Advanced")

        display(tab)

    def _create_genicam_controls(self, style):
        """Create dynamic controls from GenICam node_map features.

        Args:
            style: Widget style dict

        Returns:
            List of accordion widgets for common GenICam features
        """
        if not hasattr(self, "node_map") or not self.node_map:
            return []

        accordions = []

        # Common GenICam features (most frequently used)
        feature_groups = {
            "Image Quality": ["Gamma", "GammaEnable", "Sharpness", "Hue", "Saturation"],
            "Black & White Level": [
                "BlackLevel",
                "BlackLevelAuto",
                "WhiteBalance",
                "WhiteBalanceAuto",
            ],
            "Frame Rate": [
                "AcquisitionFrameRate",
                "AcquisitionFrameRateEnable",
                "AcquisitionFrameRateAuto",
            ],
            "Binning": ["BinningHorizontal", "BinningVertical", "BinningSelector"],
        }

        for group_name, features in feature_groups.items():
            controls = self._create_feature_controls(features, style)
            if controls:
                accordion = widgets.Accordion(children=[widgets.VBox(controls)])
                accordion.set_title(0, group_name)
                accordions.append(accordion)

        return accordions

    def _create_advanced_controls(self, style):
        """Create advanced/rarely-used controls from GenICam node_map.

        Args:
            style: Widget style dict

        Returns:
            List of accordion widgets for advanced GenICam features
        """
        if not hasattr(self, "node_map") or not self.node_map:
            return []

        accordions = []

        # Advanced/rarely-used features
        feature_groups = {
            "Trigger": [
                "TriggerMode",
                "TriggerSource",
                "TriggerActivation",
                "TriggerDelay",
                "TriggerSelector",
                "TriggerOverlap",
            ],
            "Pixel Format & Color": [
                "PixelFormat",
                "PixelSize",
                "PixelColorFilter",
                "ReverseX",
                "ReverseY",
            ],
            "Acquisition Mode": [
                "AcquisitionMode",
                "AcquisitionStart",
                "AcquisitionStop",
                "ExposureMode",
                "ExposureAuto",
            ],
            "Timing & Strobe": [
                "LineSelector",
                "LineMode",
                "LineSource",
                "CounterSelector",
                "CounterEventSource",
            ],
            "Defect Correction": ["DefectivePixelCorrection", "DefectCorrectStaticEnable"],
            "LUT & Processing": ["LUTEnable", "LUTSelector", "LUTIndex", "LUTValue"],
            "Test Patterns": ["TestPattern", "TestPatternGeneratorSelector", "TestImageSelector"],
            "Device Control": [
                "DeviceReset",
                "DeviceTemperature",
                "DeviceTemperatureSelector",
                "SensorShutterMode",
                "SensorReadoutMode",
            ],
        }

        for group_name, features in feature_groups.items():
            controls = self._create_feature_controls(features, style)
            if controls:
                accordion = widgets.Accordion(children=[widgets.VBox(controls)])
                accordion.set_title(0, group_name)
                accordions.append(accordion)

        return accordions

    def _create_feature_controls(self, features, style):
        """Create widgets for a list of GenICam features.

        Args:
            features: List of feature names
            style: Widget style dict

        Returns:
            List of widget controls
        """
        controls = []

        for feature_name in features:
            if not hasattr(self.node_map, feature_name):
                continue

            try:
                node = getattr(self.node_map, feature_name)

                # Check if readable
                if not hasattr(node, "value"):
                    continue

                # Boolean/Enable/Auto features (checkboxes or dropdowns)
                if feature_name.endswith("Enable") or feature_name.endswith("Auto"):
                    try:
                        current_val = node.value
                        # Handle both boolean and string values
                        if isinstance(current_val, str):
                            dropdown = self._create_enum_dropdown(node, feature_name, style)
                            if dropdown:
                                controls.append(dropdown)
                        else:
                            checkbox = self._create_checkbox(node, feature_name, current_val)
                            if checkbox:
                                controls.append(checkbox)
                    except Exception:
                        pass

                # Numeric features (sliders)
                elif hasattr(node, "min") and hasattr(node, "max"):
                    try:
                        slider_box = self._create_slider(node, feature_name, style)
                        if slider_box:
                            controls.append(slider_box)
                    except Exception:
                        pass

                # Enum features (dropdowns)
                else:
                    try:
                        dropdown = self._create_enum_dropdown(node, feature_name, style)
                        if dropdown:
                            controls.append(dropdown)
                    except Exception:
                        pass

            except Exception:
                pass

        return controls

    def _create_checkbox(self, node, feature_name, current_val):
        """Create checkbox widget for boolean GenICam feature."""
        checkbox = widgets.Checkbox(value=bool(current_val), description=feature_name, indent=False)

        def on_change(change):
            try:
                node.value = change["new"]
            except Exception as e:
                print(f"Error setting {feature_name}: {e}")

        checkbox.observe(on_change, names="value")
        return checkbox

    def _create_slider(self, node, feature_name, style):
        """Create slider widget for numeric GenICam feature."""
        try:
            min_val = float(node.min)
            max_val = float(node.max)
            current_val = float(node.value)

            # Determine if integer or float
            is_int = isinstance(node.value, int) or feature_name in ["BlackLevel"]

            if is_int:
                slider = widgets.IntSlider(
                    value=int(current_val),
                    min=int(min_val),
                    max=int(max_val),
                    description=f"{feature_name}:",
                    style=style,
                )
                input_widget = widgets.IntText(
                    value=int(current_val), layout=widgets.Layout(width="100px")
                )
            else:
                slider = widgets.FloatSlider(
                    value=current_val,
                    min=min_val,
                    max=max_val,
                    description=f"{feature_name}:",
                    style=style,
                    readout_format=".2f",
                )
                input_widget = widgets.FloatText(
                    value=current_val, layout=widgets.Layout(width="100px")
                )

            def on_slider_change(change):
                try:
                    node.value = change["new"]
                    input_widget.value = change["new"]
                except Exception as e:
                    print(f"Error setting {feature_name}: {e}")

            def on_input_change(change):
                slider.value = change["new"]

            slider.observe(on_slider_change, names="value")
            input_widget.observe(on_input_change, names="value")

            return widgets.HBox([slider, input_widget])

        except Exception:
            return None

    def _create_enum_dropdown(self, node, feature_name, style):
        """Create dropdown widget for enumeration GenICam feature."""
        try:
            current_val = str(node.value)

            # Try to get available options (GenICam enums)
            options = []
            if hasattr(node, "symbolics"):
                options = list(node.symbolics)
            elif feature_name.endswith("Enable"):
                options = ["On", "Off"]
            elif feature_name.endswith("Auto"):
                options = ["Off", "Once", "Continuous"]
            elif current_val:
                options = [current_val]

            if not options:
                return None

            dropdown = widgets.Dropdown(
                options=options,
                value=current_val if current_val in options else options[0],
                description=f"{feature_name}:",
                style=style,
            )

            def on_change(change):
                try:
                    node.value = change["new"]
                except Exception as e:
                    print(f"Error setting {feature_name}: {e}")

            dropdown.observe(on_change, names="value")
            return dropdown

        except Exception:
            return None

    def _apply_settings_from_kwargs(self, kwargs):
        """Apply camera settings from keyword arguments.

        Args:
            kwargs: Dictionary of parameter names and values to set.
                   Handles both standard camera attributes (exposure_time, gain)
                   and GenICam node_map features.
        """
        for param_name, value in kwargs.items():
            # Handle standard camera attributes
            if param_name == "exposure_time" or param_name == "ExposureTime":
                try:
                    self.set_exposure(value)
                    print(f"Set exposure_time = {value}")
                except Exception as e:
                    print(f"Error setting exposure_time: {e}")
                continue

            if param_name == "gain" or param_name == "Gain":
                try:
                    self.set_gain(value)
                    print(f"Set gain = {value}")
                except Exception as e:
                    print(f"Error setting gain: {e}")
                continue

            # Handle GenICam node_map features
            if hasattr(self, "node_map") and self.node_map:
                if hasattr(self.node_map, param_name):
                    try:
                        node = getattr(self.node_map, param_name)

                        # Convert string boolean representations to actual booleans
                        if isinstance(value, str):
                            if param_name.endswith("Enable") or param_name.endswith("Auto"):
                                # Check if this is actually a boolean node
                                try:
                                    current_val = node.value
                                    if isinstance(current_val, bool):
                                        # It's a boolean node, convert string to bool
                                        if value.lower() in ["on", "true", "1", "yes"]:
                                            value = True
                                        elif value.lower() in ["off", "false", "0", "no"]:
                                            value = False
                                        # else keep as string (might be enum like 'Once', 'Continuous')
                                except Exception:
                                    pass

                        node.value = value
                        print(f"Set {param_name} = {value}")
                    except Exception as e:
                        print(f"Error setting {param_name}: {e}")
                else:
                    print(f"Warning: Parameter '{param_name}' not found in node_map")
            else:
                print(f"Warning: Parameter '{param_name}' not recognized")
