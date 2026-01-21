"""Laser beam profiler with Gaussian fitting and visualization."""

import argparse
import logging
import os
import threading
import time
import traceback
import webbrowser

import numpy as np
import plotly.graph_objs as go
from PIL import Image
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

try:
    from .basler import BaslerCamera
    from .camera import Camera
    from .flir import FlirCamera
    from .simulated import SimulatedCamera
except ImportError:
    # Running as a script, use absolute imports
    import sys
    from pathlib import Path

    # Add parent directory to path
    script_dir = Path(__file__).resolve().parent
    if str(script_dir.parent) not in sys.path:
        sys.path.insert(0, str(script_dir.parent))

    from pybeamprofiler.basler import BaslerCamera
    from pybeamprofiler.camera import Camera
    from pybeamprofiler.flir import FlirCamera
    from pybeamprofiler.simulated import SimulatedCamera

logger = logging.getLogger(__name__)


class BeamProfiler:
    """Laser beam profiler with Gaussian fitting capabilities.

    Supports 1D and 2D Gaussian fitting of beam profiles from static images
    or camera streams. Provides beam width measurements in various definitions.

    Args:
        camera: Camera type ('simulated', 'flir', 'basler') or None for default
        file: Path to static image file to analyze
        fit: Fitting method ('1d', '2d', 'linecut')
        definition: Width definition ('gaussian' for 1/e², 'fwhm', 'd4s')
        exposure_time: Camera exposure time in seconds (default: camera default)
        pixel_size: Pixel size in micrometers

    Attributes:
        width_x: Beam width in x direction (μm)
        width_y: Beam width in y direction (μm)
        center_x: Beam center x position (pixels)
        center_y: Beam center y position (pixels)
        angle_deg: Beam rotation angle (degrees, for 2D fit)
        peak_value: Peak intensity value
    """

    def __init__(
        self,
        camera: str | None = None,
        file: str | None = None,
        fit: str = "1d",
        definition: str = "gaussian",
        exposure_time: float | None = None,
        pixel_size: float | None = None,
    ):
        """Initialize the beam profiler.

        Raises:
            ValueError: If neither camera nor file is provided or successfully loaded
            RuntimeError: If physical camera (FLIR/Basler) fails to open
            AssertionError: If pixel_size not provided for static image files
        """
        self.camera: Camera | None = None
        self.fit_method = fit
        self.definition = definition

        # Results
        self.width_x = 0.0
        self.width_y = 0.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.angle_deg = 0.0
        self.peak_value = 0.0

        # Caching for initial guesses
        self._last_popt_x = None
        self._last_popt_y = None
        self._last_popt_2d = None

        self.last_img = None  # Initialize before file/camera loading

        if file:
            self._load_file(file)
            self._mode = "static"
            assert pixel_size is not None, "Pixel size must be provided for static beam image files"
            self.pixel_size = pixel_size
        elif camera:
            self._initialize_camera(camera)
        else:
            self.camera = SimulatedCamera()
            self.camera.open()
            self._mode = "camera"

        if self.camera:
            self.width_pixels = self.camera.width
            self.height_pixels = self.camera.height
            self.pixel_size = self.camera.pixel_size
            # Set exposure time if provided
            if exposure_time is not None:
                self.camera.set_exposure(exposure_time)
        elif file and self.last_img is not None:
            # For static files, dimensions already set in _load_file
            pass
        else:
            raise ValueError("Either camera or file must be provided and successfully loaded")

    def _initialize_camera(self, camera: str) -> None:
        """Initialize camera hardware.

        Args:
            camera: Camera type string

        Raises:
            RuntimeError: If physical camera fails to open
        """
        camera_lower = camera.lower()
        if camera_lower == "flir":
            self.camera = FlirCamera()
        elif camera_lower == "basler":
            self.camera = BaslerCamera()
        elif camera_lower == "simulated":
            self.camera = SimulatedCamera()
        else:
            logger.warning(f"Unknown camera {camera}, using Simulated.")
            self.camera = SimulatedCamera()

        try:
            self.camera.open()
            self._mode = "camera"
        except Exception as e:
            # Don't fallback to simulated for physical cameras
            if camera_lower in ["flir", "basler"]:
                logger.error(f"Failed to open {camera} camera: {e}")
                raise RuntimeError(f"Failed to open {camera} camera: {e}") from e
            else:
                # Only fallback for unknown/simulated cameras
                logger.error(f"Failed to open camera: {e}")
                self.camera = SimulatedCamera()
                self.camera.open()
                self._mode = "camera"

    def _load_file(self, filename: str) -> None:
        """Load static image file.

        Args:
            filename: Path to image file
        """
        try:
            with Image.open(filename) as img:
                self.last_img = np.array(img)
                self.width_pixels = self.last_img.shape[1]
                self.height_pixels = self.last_img.shape[0]
        except Exception as e:
            logger.error(f"Error loading image file {filename}: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure camera is closed."""
        if self.camera:
            try:
                self.camera.close()
            except Exception as e:
                logger.warning(f"Error closing camera: {e}")
        return False

    def __getattr__(self, name: str):
        """Proxy camera attributes."""
        if self.camera and hasattr(self.camera, name):
            return getattr(self.camera, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @staticmethod
    def gaussian(x: np.ndarray, a: float, x0: float, sigma: float, offset: float) -> np.ndarray:
        """1D Gaussian function.

        Args:
            x: Input array
            a: Amplitude
            x0: Center position
            sigma: Standard deviation
            offset: Baseline offset

        Returns:
            Gaussian values at x positions
        """
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset

    @staticmethod
    def gaussian_2d(
        xy: tuple[np.ndarray, np.ndarray],
        amplitude: float,
        x0: float,
        y0: float,
        sigma_x: float,
        sigma_y: float,
        theta: float,
        offset: float,
    ) -> np.ndarray:
        """2D Gaussian function with rotation.

        Args:
            xy: tuple of (x, y) mesh grids
            amplitude: Peak amplitude
            x0: Center x position
            y0: Center y position
            sigma_x: Standard deviation in x
            sigma_y: Standard deviation in y
            theta: Rotation angle in radians
            offset: Baseline offset

        Returns:
            Flattened 2D Gaussian values
        """
        x, y = xy
        x0 = float(x0)
        y0 = float(y0)
        a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)
        g = offset + amplitude * np.exp(
            -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
        )
        return g.ravel()

    @property
    def width(self) -> float:
        """Average beam width (μm)."""
        return (self.width_x + self.width_y) / 2

    @property
    def diameter(self) -> float:
        """Beam diameter, same as width (μm)."""
        return self.width

    @property
    def radius(self) -> float:
        """Beam radius (μm)."""
        return self.width / 2

    @property
    def fwhm_x(self) -> float:
        """Full Width at Half Maximum in X direction (μm)."""
        sigma_x = self.width_x / 4.0
        return 2.355 * sigma_x

    @property
    def fwhm_y(self) -> float:
        """Full Width at Half Maximum in Y direction (μm)."""
        sigma_y = self.width_y / 4.0
        return 2.355 * sigma_y

    @property
    def fw_1e_x(self) -> float:
        """Full Width at 1/e in X direction (μm)."""
        sigma_x = self.width_x / 4.0
        return 2.0 * sigma_x

    @property
    def fw_1e_y(self) -> float:
        """Full Width at 1/e in Y direction (μm)."""
        sigma_y = self.width_y / 4.0
        return 2.0 * sigma_y

    @property
    def fw_1e2_x(self) -> float:
        """Full Width at 1/e² in X direction (μm) - same as width_x."""
        return self.width_x

    @property
    def fw_1e2_y(self) -> float:
        """Full Width at 1/e² in Y direction (μm) - same as width_y."""
        return self.width_y

    @property
    def height_x(self) -> float:
        """Peak height in X profile (intensity units)."""
        return self.peak_value

    @property
    def height_y(self) -> float:
        """Peak height in Y profile (intensity units)."""
        return self.peak_value

    def _measure_fwhm(self, profile: np.ndarray) -> tuple[float, float, float]:
        """Measure Full Width at Half Maximum directly from profile.

        Uses linear interpolation for sub-pixel accuracy without assuming
        Gaussian distribution.

        Args:
            profile: 1D intensity profile

        Returns:
            Tuple of (center, fwhm_width, peak_value)
        """
        profile = profile - np.min(profile)  # Remove baseline
        peak_idx = np.argmax(profile)
        peak_value = profile[peak_idx]
        half_max = peak_value / 2.0

        # Find left half-maximum point
        left_idx = peak_idx
        while left_idx > 0 and profile[left_idx] > half_max:
            left_idx -= 1
        # Interpolate
        if left_idx < peak_idx and profile[left_idx] < half_max:
            frac = (half_max - profile[left_idx]) / (profile[left_idx + 1] - profile[left_idx])
            left_pos = left_idx + frac
        else:
            left_pos = float(left_idx)

        # Find right half-maximum point
        right_idx = peak_idx
        while right_idx < len(profile) - 1 and profile[right_idx] > half_max:
            right_idx += 1
        # Interpolate
        if right_idx > peak_idx and profile[right_idx] < half_max:
            frac = (half_max - profile[right_idx]) / (profile[right_idx - 1] - profile[right_idx])
            right_pos = right_idx - frac
        else:
            right_pos = float(right_idx)

        fwhm = right_pos - left_pos
        center = (left_pos + right_pos) / 2.0

        return center, fwhm, peak_value

    def _measure_d4s(self, profile: np.ndarray) -> tuple[float, float]:
        """Measure D4σ (ISO 11146 second moment width) directly from profile.

        Uses intensity-weighted second moment without assuming Gaussian distribution.

        Args:
            profile: 1D intensity profile

        Returns:
            Tuple of (center, d4sigma_width)
        """
        profile = profile - np.min(profile)  # Remove baseline
        profile = np.maximum(profile, 0)  # Ensure non-negative

        total_intensity = np.sum(profile)
        if total_intensity == 0:
            return len(profile) / 2.0, 1.0

        x = np.arange(len(profile))

        # First moment (center)
        center = np.sum(x * profile) / total_intensity

        # Second moment (variance)
        variance = np.sum(((x - center) ** 2) * profile) / total_intensity
        sigma = np.sqrt(variance)

        d4sigma = 4.0 * sigma

        return center, d4sigma

    def _fit_1d_gaussian(self, profile: np.ndarray, last_popt: list | None = None) -> list:
        """Fit 1D Gaussian to profile.

        Args:
            profile: 1D intensity profile
            last_popt: Previous fit parameters for initial guess

        Returns:
            Fit parameters [amplitude, center, sigma, offset]
        """
        n = len(profile)
        if n == 0:
            return [0, 0, 1, 0]

        if last_popt is not None:
            p0 = last_popt
        else:
            pmax, pmin = np.max(profile), np.min(profile)
            p0 = [pmax - pmin, np.argmax(profile), n / 10.0, pmin]

        try:
            x = np.arange(n)
            # Bounds constrain parameters for faster convergence:
            # amplitude > 0, center in [0, n], sigma in [0.1, n], offset unbounded
            bounds = ([0, 0, 0.1, -np.inf], [np.inf, n, n, np.inf])
            popt, _ = curve_fit(BeamProfiler.gaussian, x, profile, p0=p0, bounds=bounds, maxfev=100)
            return popt
        except (RuntimeError, ValueError) as e:
            logger.warning(f"1D fit failed: {e}, using initial guess")
            return p0

    def _fit_2d_gaussian(self, image: np.ndarray) -> list:
        """Fit 2D Gaussian to image.

        Args:
            image: 2D intensity array

        Returns:
            Fit parameters [amplitude, x0, y0, sigma_x, sigma_y, theta, offset]
        """
        h, w = image.shape

        if self._last_popt_2d is not None:
            p0 = self._last_popt_2d
        else:
            pmax, pmin = np.max(image), np.min(image)
            y0, x0 = np.unravel_index(np.argmax(image), image.shape)
            p0 = [pmax - pmin, x0, y0, w / 10.0, h / 10.0, 0.0, pmin]

        try:
            x, y = np.arange(w), np.arange(h)
            xv, yv = np.meshgrid(x, y)
            # Bounds constrain parameters for faster convergence:
            # amplitude > 0, centers in image, sigmas > 0.1, theta in [-π, π]
            bounds = ([0, 0, 0, 0.1, 0.1, -np.pi, -np.inf], [np.inf, w, h, w, h, np.pi, np.inf])
            popt, _ = curve_fit(
                BeamProfiler.gaussian_2d,
                (xv.ravel(), yv.ravel()),
                image.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=100,
            )
            self._last_popt_2d = popt
            return popt
        except (RuntimeError, ValueError) as e:
            logger.warning(f"2D fit failed: {e}, using initial guess")
            return p0

    def analyze(self, image: np.ndarray) -> tuple[list | None, list | None]:
        """Analyze beam image and extract parameters.

        Args:
            image: 2D intensity array

        Returns:
            tuple of (x_fit_params, y_fit_params) for 1D projections

        Raises:
            ValueError: If image is None or not 2D
        """
        if image is None:
            raise ValueError("Image cannot be None")

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(image)}")

        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got {image.ndim}D array")

        if image.size == 0:
            raise ValueError("Image cannot be empty")

        self.peak_value = float(np.max(image))

        # Use direct measurement for FWHM and D4σ (no Gaussian assumption)
        if self.definition in ["fwhm", "d4s"]:
            proj_x = np.sum(image, axis=0)
            proj_y = np.sum(image, axis=1)

            if self.definition == "fwhm":
                center_x, width_x, _ = self._measure_fwhm(proj_x)
                center_y, width_y, _ = self._measure_fwhm(proj_y)
            else:  # d4s
                center_x, width_x = self._measure_d4s(proj_x)
                center_y, width_y = self._measure_d4s(proj_y)

            self.center_x = center_x
            self.center_y = center_y
            self.width_x = width_x * self.pixel_size
            self.width_y = width_y * self.pixel_size
            self.angle_deg = 0.0

            # Still fit Gaussian for visualization
            popt_x = self._fit_1d_gaussian(proj_x, self._last_popt_x)
            popt_y = self._fit_1d_gaussian(proj_y, self._last_popt_y)
            self._last_popt_x, self._last_popt_y = popt_x, popt_y

            return popt_x, popt_y

        # Gaussian-based fitting for 'gaussian' definition
        if self.fit_method == "linecut":
            peak_y, peak_x = np.unravel_index(np.argmax(image), image.shape)
            linecut_x = image[peak_y, :]
            linecut_y = image[:, peak_x]

            # Store linecut positions for visualization
            self._linecut_x = peak_x
            self._linecut_y = peak_y

            popt_x = self._fit_1d_gaussian(linecut_x, self._last_popt_x)
            popt_y = self._fit_1d_gaussian(linecut_y, self._last_popt_y)
            self._last_popt_x, self._last_popt_y = popt_x, popt_y

            self._update_widths(abs(popt_x[2]), abs(popt_y[2]))
            self.center_x, self.center_y = popt_x[1], popt_y[1]
            self.angle_deg = 0.0

            return popt_x, popt_y

        elif self.fit_method == "2d":
            popt = self._fit_2d_gaussian(image)
            _, x0, y0, sigma_x, sigma_y, theta, _ = popt

            self._update_widths(abs(sigma_x), abs(sigma_y))
            self.center_x, self.center_y = x0, y0
            self.angle_deg = np.degrees(theta) % 180

            proj_x = np.sum(image, axis=0)
            proj_y = np.sum(image, axis=1)
            popt_x = self._fit_1d_gaussian(proj_x, self._last_popt_x)
            popt_y = self._fit_1d_gaussian(proj_y, self._last_popt_y)
            self._last_popt_x, self._last_popt_y = popt_x, popt_y

            return popt_x, popt_y

        else:
            proj_x = np.sum(image, axis=0)
            proj_y = np.sum(image, axis=1)

            popt_x = self._fit_1d_gaussian(proj_x, self._last_popt_x)
            popt_y = self._fit_1d_gaussian(proj_y, self._last_popt_y)
            self._last_popt_x, self._last_popt_y = popt_x, popt_y

            self._update_widths(abs(popt_x[2]), abs(popt_y[2]))
            self.center_x, self.center_y = popt_x[1], popt_y[1]
            self.angle_deg = 0.0

            return popt_x, popt_y

    def _update_widths(self, sigma_x: float, sigma_y: float) -> None:
        """Update width parameters from Gaussian sigma values.

        Converts sigma to the selected width definition (gaussian/fwhm/d4s).

        Args:
            sigma_x: Gaussian sigma in x (pixels)
            sigma_y: Gaussian sigma in y (pixels)
        """
        if self.definition == "fwhm":
            factor = 2.355  # 2*sqrt(2*ln(2))
        elif self.definition == "d4s":
            factor = 4.0  # 4 sigma
        else:  # 'gaussian' - 1/e²
            factor = 4.0  # 4.0 sigma for 1/e² width
        self.width_x = factor * sigma_x * self.pixel_size
        self.width_y = factor * sigma_y * self.pixel_size

    def plot(
        self,
        num_img: int | None = None,
        heatmap_only: bool = False,
    ) -> None:
        """Display beam profile with Gaussian fitting visualization.

        Args:
            num_img: Number of images (1 for single shot, None for continuous streaming)
            heatmap_only: Show only heatmap for faster rendering (~30 Hz vs ~23 Hz full plot)
        """

        self._heatmap_only = heatmap_only  # Store for _plot_stream to use

        if num_img == 1 or self._mode == "static":
            self._plot_single()
        else:
            self._plot_stream()

    def _create_fast_figure(
        self, image: np.ndarray, popt_x: list | None, popt_y: list | None
    ) -> go.Figure:
        """Create simplified figure with heatmap only for faster rendering.

        Args:
            image: 2D intensity array
            popt_x: X projection fit parameters
            popt_y: Y projection fit parameters

        Returns:
            Plotly figure with heatmap and ellipse overlay
        """
        if image is None:
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=image, colorscale="Viridis", showscale=True))

        # Add linecut crosshair lines if using linecut method
        if (
            self.fit_method == "linecut"
            and hasattr(self, "_linecut_x")
            and hasattr(self, "_linecut_y")
        ):
            # Vertical line at linecut_x
            fig.add_trace(
                go.Scatter(
                    x=[self._linecut_x, self._linecut_x],
                    y=[0, image.shape[0] - 1],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                    name="Linecut X",
                    showlegend=False,
                )
            )
            # Horizontal line at linecut_y
            fig.add_trace(
                go.Scatter(
                    x=[0, image.shape[1] - 1],
                    y=[self._linecut_y, self._linecut_y],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                    name="Linecut Y",
                    showlegend=False,
                )
            )

        if popt_x is not None and popt_y is not None:
            cx, cy = popt_x[1], popt_y[1]
            rx, ry = 2 * abs(popt_x[2]), 2 * abs(popt_y[2])
            theta_vals = np.linspace(0, 2 * np.pi, 100)

            if self.fit_method == "2d" and hasattr(self, "angle_deg"):
                angle_rad = np.radians(self.angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                cos_t, sin_t = np.cos(theta_vals), np.sin(theta_vals)
                x_ellipse = cx + rx * cos_t * cos_a - ry * sin_t * sin_a
                y_ellipse = cy + rx * cos_t * sin_a + ry * sin_t * cos_a
            else:
                x_ellipse = cx + rx * np.cos(theta_vals)
                y_ellipse = cy + ry * np.sin(theta_vals)

            fig.add_trace(
                go.Scatter(
                    x=x_ellipse,
                    y=y_ellipse,
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name=f"{self.definition} Width",
                    showlegend=False,
                )
            )

        # Compact title
        title = f"X={self.width_x:.0f}μm Y={self.width_y:.0f}μm"
        if self.fit_method == "2d":
            title += f" θ={self.angle_deg:.0f}°"
        title += f" Peak={self.peak_value:.0f}"

        fig.update_layout(
            height=600,
            width=600,
            title_text=title,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            showlegend=False,
        )

        return fig

    def _create_figure(
        self, image: np.ndarray, popt_x: list | None, popt_y: list | None
    ) -> go.Figure:
        """Create complete figure with beam image and projection plots.

        Args:
            image: 2D intensity array
            popt_x: X projection fit parameters
            popt_y: Y projection fit parameters

        Returns:
            Plotly figure with 2D heatmap and aligned X/Y projection plots
        """
        if image is None:
            return go.Figure()

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.3, 0.7],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "heatmap"}, {"type": "xy"}],
            ],
            subplot_titles=("X Profile", "", "Beam Image", "Y Profile"),
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )

        # Beam Image (heatmap)
        fig.add_trace(go.Heatmap(z=image, colorscale="Viridis", showscale=False), row=2, col=1)

        # Add linecut crosshair lines if using linecut method
        if (
            self.fit_method == "linecut"
            and hasattr(self, "_linecut_x")
            and hasattr(self, "_linecut_y")
        ):
            # Vertical line at linecut_x
            fig.add_trace(
                go.Scatter(
                    x=[self._linecut_x, self._linecut_x],
                    y=[0, image.shape[0] - 1],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                    name="Linecut X",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )
            # Horizontal line at linecut_y
            fig.add_trace(
                go.Scatter(
                    x=[0, image.shape[1] - 1],
                    y=[self._linecut_y, self._linecut_y],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                    name="Linecut Y",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        if popt_x is not None and popt_y is not None:
            cx, cy = popt_x[1], popt_y[1]
            rx, ry = 2 * abs(popt_x[2]), 2 * abs(popt_y[2])
            theta_vals = np.linspace(0, 2 * np.pi, 100)

            if self.fit_method == "2d" and hasattr(self, "angle_deg"):
                angle_rad = np.radians(self.angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                cos_t, sin_t = np.cos(theta_vals), np.sin(theta_vals)
                x_ellipse = cx + rx * cos_t * cos_a - ry * sin_t * sin_a
                y_ellipse = cy + rx * cos_t * sin_a + ry * sin_t * cos_a
            else:
                x_ellipse = cx + rx * np.cos(theta_vals)
                y_ellipse = cy + ry * np.sin(theta_vals)

            fig.add_trace(
                go.Scatter(
                    x=x_ellipse,
                    y=y_ellipse,
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name=f"{self.definition} Width",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        # X Profile (Integrated) - Above beam image
        x = np.arange(len(image[0]))
        proj_x = np.sum(image, axis=0)
        fig.add_trace(
            go.Scatter(x=x, y=proj_x, mode="markers", name="Data X", marker=dict(size=2)),
            row=1,
            col=1,
        )
        if popt_x is not None:
            fitted_x = BeamProfiler.gaussian(x, *popt_x)
            fig.add_trace(go.Scatter(x=x, y=fitted_x, mode="lines", name="Fit X"), row=1, col=1)

        # Y Profile (Integrated) - Right of beam image, rotated
        y = np.arange(len(image))
        proj_y = np.sum(image, axis=1)
        fig.add_trace(
            go.Scatter(x=proj_y, y=y, mode="markers", name="Data Y", marker=dict(size=2)),
            row=2,
            col=2,
        )
        if popt_y is not None:
            fitted_y = BeamProfiler.gaussian(y, *popt_y)
            fig.add_trace(go.Scatter(x=fitted_y, y=y, mode="lines", name="Fit Y"), row=2, col=2)

        title = f"{self.definition.upper()} Width: X={self.width_x:.1f} μm, Y={self.width_y:.1f} μm"
        if self.fit_method == "2d":
            title += f", Angle: {self.angle_deg:.1f}°"
        title += f", Peak: {self.peak_value:.0f}"

        fig.update_layout(
            height=700,
            width=900,
            title_text=title,
            showlegend=True,
            margin=dict(l=40, r=20, t=80, b=40),
        )

        # Align X profile's x-axis with beam image's x-axis
        fig.update_xaxes(matches="x3", row=1, col=1, showticklabels=False)

        # Align Y profile's y-axis with beam image's y-axis
        fig.update_yaxes(matches="y3", row=2, col=2, showticklabels=False)

        # Ensure proper aspect ratio for beam image
        fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=2, col=1)
        fig.update_xaxes(constrain="domain", row=2, col=1)

        return fig

    def _plot_single(self) -> None:
        """Capture and plot single image."""
        if self._mode == "camera":
            self.camera.start_acquisition()
            img = self.camera.get_image()
            self.camera.stop_acquisition()
        else:
            img = self.last_img

        popt_x, popt_y = self.analyze(img)
        fig = self._create_figure(img, popt_x, popt_y)
        fig.show()

    def _plot_stream(self) -> None:
        """Start continuous streaming with live updates."""

        # Ensure camera is ready for continuous acquisition
        if self._mode == "camera":
            if not self.camera.is_acquiring:
                self.camera.start_acquisition()

        # Check for heatmap only mode
        heatmap_only = getattr(self, "_heatmap_only", False)

        try:
            # Check if running in Jupyter
            from IPython.display import clear_output, display

            get_ipython()

            # Use clear_output for live updates
            if heatmap_only:
                logger.info("Starting live stream (heatmap only, ~25-30 Hz)...")
            else:
                logger.info("Starting live stream (~6-10 Hz)...")
            logger.info("Press Jupyter's interrupt button (■) to stop\n")

            frame_count = 0
            start_time = time.time()

            try:
                while True:
                    # Get and analyze image
                    img = self.camera.get_image() if self._mode == "camera" else self.last_img
                    if img is None:
                        break

                    # Perform Gaussian fitting
                    popt_x, popt_y = self.analyze(img)

                    # Create figure
                    if heatmap_only:
                        fig = self._create_fast_figure(img, popt_x, popt_y)
                    else:
                        fig = self._create_figure(img, popt_x, popt_y)

                    # Add frame info to title
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0

                    current_title = fig.layout.title.text if fig.layout.title else ""
                    fig.update_layout(
                        title_text=f"{current_title}<br><sub>Frame #{frame_count} | FPS: {fps:.1f}</sub>"
                    )

                    # Clear and display updated figure
                    clear_output(wait=True)
                    display(fig)

            except KeyboardInterrupt:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"\nStream stopped: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps)"
                )

        except (NameError, ImportError):
            # Running from command line - use Dash
            try:
                import dash
                from dash import dcc, html
                from dash.dependencies import Input, Output
            except ImportError:
                logger.info("\nDash not available. Using matplotlib fallback.")
                logger.info("Install dash for better performance: pip install dash\n")

                # Matplotlib fallback
                try:
                    import matplotlib.pyplot as plt
                    from matplotlib.animation import FuncAnimation

                    fig_plt, axes = plt.subplots(2, 2, figsize=(10, 8))
                    fig_plt.tight_layout(pad=3.0)

                    def update_frame(frame_num):
                        img = self.camera.get_image() if self._mode == "camera" else self.last_img
                        if img is None:
                            return

                        popt_x, popt_y = self.analyze(img)

                        # Clear all axes
                        for ax in axes.flat:
                            ax.clear()

                        # Beam image
                        axes[1, 0].imshow(img, cmap="viridis")
                        axes[1, 0].set_title("Beam Image")
                        axes[1, 0].set_xlabel("X (pixels)")
                        axes[1, 0].set_ylabel("Y (pixels)")

                        # Add ellipse overlay
                        if popt_x is not None and popt_y is not None:
                            from matplotlib.patches import Ellipse

                            cx, cy = popt_x[1], popt_y[1]
                            width_px, height_px = 4 * abs(popt_x[2]), 4 * abs(popt_y[2])
                            ellipse = Ellipse(
                                (cx, cy),
                                width_px,
                                height_px,
                                fill=False,
                                edgecolor="red",
                                linewidth=2,
                                linestyle="--",
                            )
                            axes[1, 0].add_patch(ellipse)

                        # X profile
                        x = np.arange(img.shape[1])
                        proj_x = np.sum(img, axis=0)
                        axes[0, 0].plot(x, proj_x, "o", markersize=2, label="Data")
                        if popt_x is not None:
                            fitted_x = BeamProfiler.gaussian(x, *popt_x)
                            axes[0, 0].plot(x, fitted_x, "r-", label="Fit")
                        axes[0, 0].set_title("X Profile")
                        axes[0, 0].set_xlabel("X (pixels)")
                        axes[0, 0].legend()

                        # Y profile
                        y = np.arange(img.shape[0])
                        proj_y = np.sum(img, axis=1)
                        axes[1, 1].plot(proj_y, y, "o", markersize=2, label="Data")
                        if popt_y is not None:
                            fitted_y = BeamProfiler.gaussian(y, *popt_y)
                            axes[1, 1].plot(fitted_y, y, "r-", label="Fit")
                        axes[1, 1].set_title("Y Profile")
                        axes[1, 1].set_ylabel("Y (pixels)")
                        axes[1, 1].invert_xaxis()
                        axes[1, 1].legend()

                        # Info panel
                        axes[0, 1].axis("off")
                        info_text = f"Frame: {frame_num}\n\n"
                        info_text += f"Width X: {self.width_x:.1f} μm\n"
                        info_text += f"Width Y: {self.width_y:.1f} μm\n"
                        info_text += f"Center: ({self.center_x:.1f}, {self.center_y:.1f})\n"
                        if self.fit_method == "2d":
                            info_text += f"Angle: {self.angle_deg:.1f}°\n"
                        info_text += f"Peak: {self.peak_value:.0f}"
                        axes[0, 1].text(
                            0.1,
                            0.5,
                            info_text,
                            fontsize=12,
                            verticalalignment="center",
                            family="monospace",
                        )

                    logger.info("Starting matplotlib animation (press Ctrl+C to stop)...")
                    _anim = FuncAnimation(
                        fig_plt, update_frame, interval=50, cache_frame_data=False
                    )
                    plt.show()

                except ImportError:
                    logger.info("ERROR: Neither dash nor matplotlib is installed.")
                    logger.info("   Install one of them:")
                    logger.info("   - pip install dash (recommended for streaming)")
                    logger.info("   - pip install matplotlib")
                    return

                return

            # Dash is available, use it
            app = dash.Dash(__name__)

            # Flag to signal shutdown
            shutdown_flag = {"stop": False}

            # Start acquisition before Dash server
            if self._mode == "camera" and not self.camera.is_acquiring:
                self.camera.start_acquisition()

            # Get initial image for display
            initial_img = self.camera.get_image() if self._mode == "camera" else self.last_img
            initial_popt_x, initial_popt_y = (
                self.analyze(initial_img) if initial_img is not None else (None, None)
            )
            initial_fig = (
                self._create_figure(initial_img, initial_popt_x, initial_popt_y)
                if initial_img is not None
                else go.Figure()
            )

            app.layout = html.Div(
                [
                    dcc.Graph(
                        id="live-update-graph",
                        figure=initial_fig,
                        style={"height": "700px"},
                    ),
                    dcc.Interval(
                        id="interval-component",
                        interval=100,  # Update every 100ms (10 Hz) for reliable streaming
                        n_intervals=0,
                    ),
                ]
            )

            @app.callback(
                Output("live-update-graph", "figure"),
                Input("interval-component", "n_intervals"),
            )
            def update_graph_live(n):
                # Check if shutdown requested
                if shutdown_flag["stop"]:
                    return go.Figure()

                if n % 10 == 0:
                    logger.debug(f"Processing frame {n}")

                if self._mode == "camera" and not self.camera.is_acquiring:
                    try:
                        self.camera.start_acquisition()
                    except Exception:
                        return go.Figure()

                try:
                    img = self.camera.get_image() if self._mode == "camera" else self.last_img
                except Exception as e:
                    # Camera fetch failed (likely stopped), return empty figure
                    logger.debug(f"Failed to get image: {e}")
                    return go.Figure()

                if img is None:
                    return go.Figure()

                popt_x, popt_y = self.analyze(img)

                # Use heatmap_only mode if set
                if heatmap_only:
                    fig = self._create_fast_figure(img, popt_x, popt_y)
                else:
                    fig = self._create_figure(img, popt_x, popt_y)

                # Always add frame number to show updates
                current_title = fig.layout.title.text if fig.layout.title else ""
                fig.update_layout(title_text=f"{current_title} | Frame #{n}")

                return fig

            logger.info("Starting Dash server at http://127.0.0.1:8050")
            logger.info("Opening browser automatically...")
            logger.info("Press Ctrl+C to stop\n")

            # Suppress Flask/Werkzeug logs for cleaner output
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            logging.getLogger("dash").setLevel(logging.ERROR)

            # Open browser automatically after a short delay (unless disabled for testing)
            if os.environ.get("PYBEAMPROFILER_NO_BROWSER") != "1":

                def open_browser():
                    webbrowser.open("http://127.0.0.1:8050")

                threading.Thread(target=open_browser, daemon=True).start()

            try:
                app.run(debug=False, port=8050)
            except KeyboardInterrupt:
                logger.info("\n\nStopping Dash server...")
                shutdown_flag["stop"] = True


if __name__ == "__main__":
    """Main entry point for command-line interface."""

    parser = argparse.ArgumentParser(
        description="pyBeamprofiler - Laser beam profiler with Gaussian fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Simulated camera with continuous streaming
        python -m pybeamprofiler.beamprofiler

        # FLIR camera, single shot
        python -m pybeamprofiler.beamprofiler --camera flir --num-img 1

        # Static image file
        python -m pybeamprofiler.beamprofiler --file beam.png

        # Basler camera with 2D fitting and FWHM definition
        python -m pybeamprofiler.beamprofiler --camera basler --fit 2d --definition fwhm

        # Fast mode (heatmap only)
        python -m pybeamprofiler.beamprofiler --heatmap-only
        """,
    )

    # BeamProfiler arguments
    parser.add_argument(
        "--camera",
        type=str,
        default="simulated",
        choices=["simulated", "flir", "basler"],
        help="Camera type (default: simulated)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to static image file (overrides --camera)",
    )
    parser.add_argument(
        "--fit",
        type=str,
        default="1d",
        choices=["1d", "2d", "linecut"],
        help="Fitting method: 1d (fastest), 2d (with rotation), linecut (default: 1d)",
    )
    parser.add_argument(
        "--definition",
        type=str,
        default="gaussian",
        choices=["gaussian", "fwhm", "d4s"],
        help="Width definition: gaussian (1/e²), fwhm, d4s (default: gaussian)",
    )
    parser.add_argument(
        "--exposure-time",
        type=float,
        default=None,
        help="Camera exposure time in seconds (set during initialization)",
    )

    # plot() arguments
    parser.add_argument(
        "--num-img",
        type=int,
        default=None,
        help="Number of images: 1 for single shot, None for continuous (default: continuous)",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Show only heatmap for faster display (~8-12 Hz in Jupyter)",
    )

    # Additional options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Create BeamProfiler instance
    logger.info("Initializing pyBeamprofiler...")
    logger.info(f"   Camera: {args.file if args.file else args.camera}")
    logger.info(f"   Fitting: {args.fit} ({args.definition})")

    bp = BeamProfiler(
        camera=None if args.file else args.camera,
        file=args.file,
        fit=args.fit,
        definition=args.definition,
        exposure_time=args.exposure_time,
    )

    logger.info(f"   Sensor: {bp.width_pixels}×{bp.height_pixels} pixels")
    logger.info(f"   Pixel size: {bp.pixel_size:.2f} μm")

    # Start acquisition
    if args.num_img == 1:
        logger.info("Single shot acquisition...")
    else:
        logger.info("Starting continuous streaming...")
        logger.info("   Press Ctrl+C to stop")

    try:
        bp.plot(num_img=args.num_img, heatmap_only=args.heatmap_only)
    except Exception as e:
        logger.info(f"\nERROR: {e}")
        if args.verbose:
            traceback.print_exc()
    finally:
        # Ensure camera is properly closed
        if hasattr(bp, "camera") and bp.camera:
            try:
                if bp.camera.is_acquiring:
                    bp.camera.stop_acquisition()
                    logger.info("Camera acquisition stopped")
                bp.camera.close()
                logger.info("Camera closed")
            except Exception:
                pass
