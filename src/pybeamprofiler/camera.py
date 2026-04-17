"""Base camera interface for beam profiler."""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Features handled by dedicated exposure / gain / ROI controls.
_HANDLED_FEATURES = frozenset(
    {
        "ExposureTime",
        "ExposureTimeAbs",
        "ExposureTimeRaw",
        "Gain",
        "GainRaw",
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
        "WidthMax",
        "HeightMax",
    }
)

# Maps a feature-name prefix to an SFNC-standard category label.
# The names mirror the categories shown in SpinView / pylon Viewer.
_CATEGORY_MAP: dict[str, str] = {
    # Acquisition Control
    "Acquisition": "Acquisition Control",
    "Trigger": "Acquisition Control",
    "Exposure": "Acquisition Control",
    # Analog Control
    "Gain": "Analog Control",
    "BlackLevel": "Analog Control",
    "WhiteBalance": "Analog Control",
    "Balance": "Analog Control",
    "Gamma": "Analog Control",
    "Sharpness": "Analog Control",
    "Hue": "Analog Control",
    "Saturation": "Analog Control",
    # Image Format Control
    "Pixel": "Image Format Control",
    "Binning": "Image Format Control",
    "Decimation": "Image Format Control",
    "Reverse": "Image Format Control",
    "Sensor": "Image Format Control",
    # Device Control
    "Device": "Device Control",
    # Digital I/O Control
    "Line": "Digital I/O Control",
    "UserOutput": "Digital I/O Control",
    # Counter & Timer Control
    "Counter": "Counter & Timer Control",
    "Timer": "Counter & Timer Control",
    # Other standard categories
    "LUT": "LUT Control",
    "Defect": "Image Quality Control",
    "Test": "Test Control",
    "Chunk": "Chunk Data Control",
    "Event": "Event Control",
}


def _categorize_feature(name: str) -> str:
    """Infer a UI group name from a GenICam feature name."""
    prefixes = cast("list[str]", sorted(_CATEGORY_MAP.keys(), key=len, reverse=True))
    for prefix in prefixes:
        if name.startswith(prefix):
            return _CATEGORY_MAP[prefix]
    match = re.match(r"([A-Z][a-z]+)", name)
    if match:
        return match.group(1)
    return "Other"


class Camera(ABC):
    """Abstract base class for camera interfaces.

    Defines the common interface for all camera types (simulated,
    FLIR, Basler, etc.).

    Attributes:
        exposure_time: Current exposure time in seconds.
        gain: Current gain value.
        is_acquiring: Whether the camera is actively acquiring images.
        width: Sensor width in pixels.
        height: Sensor height in pixels.
        pixel_size: Pixel pitch in micrometers.
        image_buffer: Last captured image, or ``None``.
    """

    def __init__(self) -> None:
        self.exposure_time: float = 0.01
        self.gain: float = 0.0
        self.is_acquiring: bool = False
        self.width: int = 0
        self.height: int = 0
        self.pixel_size: float = 1.0
        self.image_buffer: np.ndarray | None = None

    @abstractmethod
    def open(self) -> None:
        """Open connection to the camera."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close connection to the camera."""
        ...

    @abstractmethod
    def start_acquisition(self) -> None:
        """Start image acquisition."""
        ...

    @abstractmethod
    def stop_acquisition(self) -> None:
        """Stop image acquisition."""
        ...

    @abstractmethod
    def get_image(self, timeout: float | None = None) -> np.ndarray:
        """Return the most recent frame from the camera.

        Args:
            timeout: Optional maximum seconds to wait for the first frame
                during cold start. Implementations that always have a frame
                available (e.g. simulated cameras) may ignore this argument.
        """
        ...

    @abstractmethod
    def set_exposure(self, exposure_time: float) -> None:
        """Set exposure time in seconds."""
        ...

    @abstractmethod
    def set_gain(self, gain: float) -> None:
        """Set gain."""
        ...

    def setting(self, **kwargs: Any) -> None:
        """Display interactive camera controls in Jupyter Notebook.

        Creates a tabbed interface with exposure, gain, and acquisition controls.
        Dynamically populates controls from the GenICam ``node_map`` for real cameras.

        Args:
            **kwargs: Camera parameters to apply before showing the UI.
                Parameter names should match ``node_map`` feature names
                (e.g. ``ExposureTime=0.01``, ``Gain=10.0``, ``BlackLevel=0``).
        """
        import ipywidgets as widgets
        from IPython.display import display

        if kwargs:
            self._apply_settings_from_kwargs(kwargs)

        style = {"description_width": "initial"}

        exposure_min, exposure_max = 1e-6, 1.0
        if hasattr(self, "exposure_range"):
            exposure_min, exposure_max = cast(tuple[float, float], self.exposure_range)

        gain_min, gain_max = 0.0, 24.0
        if hasattr(self, "gain_range"):
            gain_min, gain_max = cast(tuple[float, float], self.gain_range)

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

        def on_exposure_change(change: dict[str, Any]) -> None:
            self.set_exposure(change["new"])
            exposure_input.value = change["new"]

        def on_gain_change(change: dict[str, Any]) -> None:
            self.set_gain(change["new"])
            gain_input.value = change["new"]

        def on_exposure_input_change(change: dict[str, Any]) -> None:
            exposure_slider.value = change["new"]

        def on_gain_input_change(change: dict[str, Any]) -> None:
            gain_slider.value = change["new"]

        def on_start_click(b: Any) -> None:
            self.start_acquisition()
            start_button.disabled = True
            stop_button.disabled = False

        def on_stop_click(b: Any) -> None:
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

        genicam_controls = self._create_genicam_controls(style)
        advanced_controls = self._create_advanced_controls(style)

        camera_info: list[Any] = []
        camera_info.append(widgets.HTML(f"<b>Camera Type:</b> {type(self).__name__}"))
        camera_info.append(widgets.HTML(f"<b>Sensor Size:</b> {self.width}×{self.height} pixels"))
        camera_info.append(widgets.HTML(f"<b>Pixel Size:</b> {self.pixel_size:.2f} μm"))

        if hasattr(self, "node_map") and self.node_map:
            try:
                if hasattr(self.node_map, "SensorDescription"):
                    desc = self.node_map.SensorDescription.value  # ty:ignore[unresolved-attribute]
                    camera_info.append(widgets.HTML(f"<b>Sensor:</b> {desc}"))
            except Exception as e:
                logger.debug(f"Optional feature SensorDescription not available: {e}")
            try:
                if hasattr(self.node_map, "DeviceModelName"):
                    model = self.node_map.DeviceModelName.value  # ty:ignore[unresolved-attribute]
                    camera_info.append(widgets.HTML(f"<b>Model:</b> {model}"))
            except Exception as e:
                logger.debug(f"Optional feature DeviceModelName not available: {e}")

        camera_info_box = widgets.VBox(camera_info)

        roi_controls: Any = None
        if hasattr(self, "roi_info") and hasattr(self, "set_roi"):
            roi = cast(dict[str, Any], self.roi_info)

            offset_x_input = widgets.IntText(
                value=roi["offset_x"],
                description="Offset X:",
                style=style,
            )
            offset_y_input = widgets.IntText(
                value=roi["offset_y"],
                description="Offset Y:",
                style=style,
            )
            width_input = widgets.IntText(value=roi["width"], description="Width:", style=style)
            height_input = widgets.IntText(value=roi["height"], description="Height:", style=style)

            roi_button = widgets.Button(
                description="Apply ROI", button_style="primary", icon="check"
            )

            roi_reset_button = widgets.Button(
                description="Full Sensor", button_style="info", icon="arrows-alt"
            )

            def on_roi_apply(b: Any) -> None:
                try:
                    self.set_roi(
                        offset_x_input.value,
                        offset_y_input.value,
                        width_input.value,
                        height_input.value,
                    )  # ty:ignore[call-non-callable]
                    updated_roi = cast(dict[str, Any], self.roi_info)
                    camera_info[
                        1
                    ].value = (
                        f"<b>Sensor Size:</b> {updated_roi['width']}×{updated_roi['height']} pixels"
                    )
                except Exception as e:
                    logger.error(f"Error setting ROI: {e}")

            def on_roi_reset(b: Any) -> None:
                try:
                    roi_max = cast(dict[str, Any], self.roi_info)
                    self.set_roi(0, 0, roi_max["max_width"], roi_max["max_height"])  # ty:ignore[call-non-callable]
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
                    logger.error(f"Error resetting ROI: {e}")

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

        settings_children = [timing_accordion, analog_accordion]

        if genicam_controls:
            settings_children.extend(genicam_controls)

        if roi_controls:
            roi_accordion = widgets.Accordion(children=[roi_controls])
            roi_accordion.set_title(0, "Region of Interest")
            settings_children.append(roi_accordion)

        info_accordion = widgets.Accordion(children=[camera_info_box])
        info_accordion.set_title(0, "Camera Information")

        tab = widgets.Tab()
        tab_children = [
            widgets.VBox(settings_children),
            info_accordion,
        ]

        if advanced_controls:
            advanced_tab = widgets.VBox(advanced_controls)
            tab_children.append(advanced_tab)

        tab.children = tab_children
        tab.set_title(0, "Camera Settings")
        tab.set_title(1, "Info")
        if advanced_controls:
            tab.set_title(2, "Advanced")

        display(tab)

    def _discover_features(self) -> dict[str, list[str]]:
        """Discover available GenICam features from the ``node_map``.

        Uses ``node_map.nodes`` (the canonical GenICam enumeration API)
        when available, falling back to ``dir()`` for simulated node maps.
        Nodes are filtered by interface type and visibility so that only
        user-facing value nodes are returned.

        Returns:
            Mapping from category name to a sorted list of feature names.
        """
        if not hasattr(self, "node_map") or not self.node_map:
            return {}

        # ── Import GenICam helpers ────────────────────────────────────
        # Allowlist: only these interface types represent user-settable
        # value nodes that can be rendered as UI controls.
        allow_iface: set[int] = set()
        invisible_threshold: int | None = None
        try:
            from genicam.genapi import EInterfaceType, EVisibility

            allow_iface = {
                int(EInterfaceType.intfIFloat),
                int(EInterfaceType.intfIInteger),
                int(EInterfaceType.intfIBoolean),
                int(EInterfaceType.intfIEnumeration),
                int(EInterfaceType.intfIString),
            }
            invisible_threshold = int(EVisibility.Invisible)
        except ImportError:
            pass

        # ── Enumerate candidate node names ────────────────────────────
        # Prefer node_map.nodes (reliable for real GenICam cameras)
        # because dir() may not list dynamically-resolved feature names.
        # We use the *node object* from the list for type / visibility
        # checks because get_node() may not preserve the derived type.
        node_names: list[str] = []
        used_nodes_api = False
        try:
            raw_nodes = getattr(self.node_map, "nodes", None)
            if raw_nodes is not None:
                for node_obj in raw_nodes:
                    try:
                        # The SWIG binding may report intfIValue(0) for all
                        # nodes in the list.  When that happens the allowlist
                        # filter is useless, so we check each node's *actual*
                        # type by probing for a ``.value`` attribute instead.
                        if allow_iface:
                            iface = getattr(node_obj, "principal_interface_type", None)
                            if iface is not None and int(iface) not in allow_iface:
                                # intfIValue(0) is the generic base; let it through
                                # and filter later via the .value probe.
                                if int(iface) != 0:
                                    continue
                        if invisible_threshold is not None:
                            vis = getattr(node_obj, "visibility", None)
                            if vis is not None and int(vis) >= invisible_threshold:
                                continue
                        nm = getattr(node_obj, "name", None)
                        if nm and isinstance(nm, str):
                            node_names.append(nm)
                    except Exception:
                        continue
                used_nodes_api = True
        except Exception as exc:
            logger.warning("node_map.nodes failed: %s", exc)

        if not node_names:
            node_names = sorted(dir(self.node_map))

        logger.debug(
            "_discover_features: %d candidates via %s (node_map type: %s)",
            len(node_names),
            ".nodes" if used_nodes_api else "dir()",
            type(self.node_map).__name__,
        )

        # ── Filter & categorise ───────────────────────────────────────
        # We cannot rely on isinstance() for nodes fetched via get_node()
        # because the SWIG binding often returns a base INode regardless of
        # the real C++ type.  Instead we probe for ``.value``: nodes that
        # have it are settable value features; those that raise (categories,
        # commands, registers) are skipped.
        groups: dict[str, list[str]] = {}
        for name in sorted(set(node_names)):
            if name.startswith("_") or name in _HANDLED_FEATURES:
                continue

            try:
                attr = getattr(self.node_map, name)
            except Exception:
                continue

            if attr is None:
                continue

            if isinstance(attr, (str, int, float, bool, bytes)):
                continue

            # Probe for .value — the definitive test for a value feature.
            # Categories, commands, registers, and ports will either lack
            # the attribute or raise when it is accessed.
            try:
                _ = attr.value
            except Exception:
                continue

            category = _categorize_feature(name)
            groups.setdefault(category, []).append(name)

        logger.debug(
            "_discover_features: %d features in %d categories: %s",
            sum(len(v) for v in groups.values()),
            len(groups),
            {k: v for k, v in groups.items()},
        )
        return groups

    def _create_genicam_controls(self, style: dict[str, str]) -> list[Any]:
        """Create dynamic controls for all discovered GenICam features.

        Uses :meth:`_discover_features` to introspect the ``node_map``
        and automatically generate widgets for every available feature.

        Args:
            style: Widget style dict (e.g. ``{"description_width": "initial"}``)

        Returns:
            List of accordion widgets grouped by feature category
        """
        import ipywidgets as widgets

        if not hasattr(self, "node_map") or not self.node_map:
            return []

        discovered = self._discover_features()
        if not discovered:
            return []

        accordions = []
        for group_name, features in sorted(discovered.items()):
            controls = self._create_feature_controls(features, style)
            if controls:
                accordion = widgets.Accordion(children=[widgets.VBox(controls)])
                accordion.set_title(0, group_name)
                accordions.append(accordion)

        return accordions

    def _create_advanced_controls(self, style: dict[str, str]) -> list[Any]:
        """Create advanced/rarely-used controls from GenICam ``node_map``.

        All features are now auto-discovered by
        :meth:`_create_genicam_controls`, so this method returns an empty
        list.  It is retained for backward compatibility.

        Args:
            style: Widget style dict

        Returns:
            Empty list
        """
        return []

    def _create_feature_controls(self, features: list[str], style: dict[str, str]) -> list[Any]:
        """Create widgets for a list of GenICam features.

        Args:
            features: Feature names to look up in the ``node_map``
            style: Widget style dict

        Returns:
            List of widget controls
        """
        controls = []

        for feature_name in features:
            if not hasattr(self.node_map, feature_name):  # ty:ignore[unresolved-attribute]
                continue

            try:
                node = getattr(self.node_map, feature_name)  # ty:ignore[unresolved-attribute]

                if not hasattr(node, "value"):
                    continue

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
                    except Exception as e:
                        logger.debug(f"Could not create checkbox for {feature_name}: {e}")

                elif hasattr(node, "min") and hasattr(node, "max"):
                    try:
                        slider_box = self._create_slider(node, feature_name, style)
                        if slider_box:
                            controls.append(slider_box)
                    except Exception as e:
                        logger.debug(f"Could not create slider for {feature_name}: {e}")

                else:
                    try:
                        dropdown = self._create_enum_dropdown(node, feature_name, style)
                        if dropdown:
                            controls.append(dropdown)
                    except Exception as e:
                        logger.debug(f"Could not create dropdown for {feature_name}: {e}")

            except Exception:
                pass

        return controls

    def _create_checkbox(self, node: Any, feature_name: str, current_val: bool) -> Any:
        """Create checkbox widget for a boolean GenICam feature."""
        import ipywidgets as widgets

        checkbox = widgets.Checkbox(value=bool(current_val), description=feature_name, indent=False)

        def on_change(change: dict[str, Any]) -> None:
            try:
                node.value = change["new"]
            except Exception as e:
                logger.error(f"Error setting {feature_name}: {e}")

        checkbox.observe(on_change, names="value")
        return checkbox

    def _create_slider(self, node: Any, feature_name: str, style: dict[str, str]) -> Any | None:
        """Create slider widget for a numeric GenICam feature."""
        import ipywidgets as widgets

        try:
            min_val = float(node.min)
            max_val = float(node.max)
            current_val = float(node.value)

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

            def on_slider_change(change: dict[str, Any]) -> None:
                try:
                    node.value = change["new"]
                    input_widget.value = change["new"]
                except Exception as e:
                    logger.error(f"Error setting {feature_name}: {e}")

            def on_input_change(change: dict[str, Any]) -> None:
                slider.value = change["new"]

            slider.observe(on_slider_change, names="value")
            input_widget.observe(on_input_change, names="value")

            return widgets.HBox([slider, input_widget])

        except Exception:
            return None

    def _create_enum_dropdown(
        self, node: Any, feature_name: str, style: dict[str, str]
    ) -> Any | None:
        """Create dropdown widget for an enumeration GenICam feature."""
        import ipywidgets as widgets

        try:
            current_val = str(node.value)

            options: list[str] = []
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

            def on_change(change: dict[str, Any]) -> None:
                try:
                    node.value = change["new"]
                except Exception as e:
                    logger.error(f"Error setting {feature_name}: {e}")

            dropdown.observe(on_change, names="value")
            return dropdown

        except Exception:
            return None

    def _apply_settings_from_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Apply camera settings from keyword arguments.

        Handles both standard camera attributes (``exposure_time``, ``gain``)
        and GenICam ``node_map`` features.

        Args:
            kwargs: Mapping of parameter names to values.
        """
        for param_name, value in kwargs.items():
            if param_name in ("exposure_time", "ExposureTime"):
                try:
                    self.set_exposure(value)
                    logger.info(f"Set exposure_time = {value}")
                except Exception as e:
                    logger.error(f"Error setting exposure_time: {e}")
                continue

            if param_name in ("gain", "Gain"):
                try:
                    self.set_gain(value)
                    logger.info(f"Set gain = {value}")
                except Exception as e:
                    logger.error(f"Error setting gain: {e}")
                continue

            if hasattr(self, "node_map") and self.node_map:
                if hasattr(self.node_map, param_name):
                    try:
                        node = getattr(self.node_map, param_name)

                        if isinstance(value, str):
                            if param_name.endswith("Enable") or param_name.endswith("Auto"):
                                # Check if this is actually a boolean node
                                try:
                                    current_val = node.value
                                    if isinstance(current_val, bool):
                                        if value.lower() in ["on", "true", "1", "yes"]:
                                            value = True
                                        elif value.lower() in ["off", "false", "0", "no"]:
                                            value = False
                                except Exception as e:
                                    logger.debug(
                                        f"Could not check boolean type for {param_name}: {e}"
                                    )

                        node.value = value
                        logger.info(f"Set {param_name} = {value}")
                    except Exception as e:
                        logger.error(f"Error setting {param_name}: {e}")
                else:
                    logger.warning(f"Parameter '{param_name}' not found in node_map")
            else:
                logger.warning(f"Parameter '{param_name}' not recognized")
