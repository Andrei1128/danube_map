import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import json
from tile_generator import BathymetryTiler, TilerConfig


class InputValidator:
    """Helper class for input validation"""

    @staticmethod
    def validate_float(value, min_val=None, max_val=None, allow_empty=False):
        """Validate float input with optional range checking"""
        if allow_empty and (value == "" or value is None):
            return True, ""

        try:
            float_val = float(value)
            if min_val is not None and float_val < min_val:
                return False, f"Value must be >= {min_val}"
            if max_val is not None and float_val > max_val:
                return False, f"Value must be <= {max_val}"
            return True, ""
        except (ValueError, TypeError):
            return False, "Must be a valid number"

    @staticmethod
    def validate_int(value, min_val=None, max_val=None, allow_empty=False):
        """Validate integer input with optional range checking"""
        if allow_empty and (value == "" or value is None):
            return True, ""

        try:
            int_val = int(value)
            if min_val is not None and int_val < min_val:
                return False, f"Value must be >= {min_val}"
            if max_val is not None and int_val > max_val:
                return False, f"Value must be <= {max_val}"
            return True, ""
        except (ValueError, TypeError):
            return False, "Must be a valid integer"

    @staticmethod
    def validate_file_path(path, must_exist=True, extension=None):
        """Validate file path"""
        if not path or path.strip() == "":
            return False, "File path cannot be empty"

        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            return False, "File does not exist"

        if extension and not path.lower().endswith(extension.lower()):
            return False, f"File must have {extension} extension"

        return True, ""

    @staticmethod
    def validate_directory_path(path, must_exist=False, create_if_missing=True):
        """Validate directory path"""
        if not path or path.strip() == "":
            return False, "Directory path cannot be empty"

        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            return False, "Directory does not exist"

        if not path_obj.exists() and create_if_missing:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                return True, "Directory created"
            except Exception as e:
                return False, f"Cannot create directory: {e}"

        if path_obj.exists() and not path_obj.is_dir():
            return False, "Path exists but is not a directory"

        return True, ""


class ValidationDialog:
    """Dialog for showing validation errors"""

    def __init__(self, parent, title, validation_errors, allow_proceed=True):
        self.result = False

        # Create top-level window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x400")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() - self.dialog.winfo_width()) // 2
        y = (self.dialog.winfo_screenheight() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")

        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Show validation errors header
        ttk.Label(main_frame, text="Validation Issues Found:",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Create scrollable text widget for errors
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        error_text = tk.Text(text_frame, height=15, wrap=tk.WORD, background="lightyellow")
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=error_text.yview)
        error_text.configure(yscrollcommand=scrollbar.set)

        error_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add errors to text widget
        for i, error in enumerate(validation_errors, 1):
            error_text.insert(tk.END, f"{i}. {error}\n")

        error_text.configure(state=tk.DISABLED)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        if allow_proceed:
            # Show both Fix and Proceed buttons
            ttk.Button(button_frame, text="Close",
                      command=self._fix_issues).pack(side=tk.RIGHT, padx=(5, 0))
            ttk.Button(button_frame, text="Proceed Anyway",
                      command=self._proceed).pack(side=tk.RIGHT)
        else:
            # Only show Close button
            ttk.Button(button_frame, text="Close",
                      command=self._fix_issues).pack(side=tk.RIGHT)

        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._fix_issues)

        # Wait for dialog to close
        self.dialog.wait_window()

    def _proceed(self):
        self.result = True
        self.dialog.destroy()

    def _fix_issues(self):
        self.result = False
        self.dialog.destroy()


class ValidatedEntry(ttk.Frame):
    """Entry widget with validation but no inline error display"""

    def __init__(self, parent, textvariable, validator_func=None, width=15, **kwargs):
        super().__init__(parent)

        self.textvariable = textvariable
        self.validator_func = validator_func
        self.last_error_msg = ""

        # Create entry widget (no error label)
        self.entry = ttk.Entry(self, textvariable=textvariable, width=width, **kwargs)
        self.entry.pack(side=tk.LEFT)

    def _validate(self, event=None):
        """Validate input silently"""
        if not self.validator_func:
            return True

        value = self.textvariable.get()
        is_valid, error_msg = self.validator_func(value)

        self.last_error_msg = error_msg

        # No visual feedback - errors only shown in dialog
        return is_valid


    def get_error_message(self):
        """Get the last error message"""
        return self.last_error_msg

    def force_validate(self):
        """Force validation and return status"""
        return self._validate()


class TileGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bathymetry Tile Generator")
        self.root.geometry("850x900")

        self.csv_file = tk.StringVar()
        self.output_dir = tk.StringVar(value="tiles")

        # Configuration variables
        self.tile_size = tk.DoubleVar(value=0.01)
        self.overlap_factor = tk.DoubleVar(value=0.1)
        self.resolution = tk.IntVar(value=1000)
        self.batch_size = tk.IntVar(value=10)
        self.smoothing_sigma = tk.DoubleVar(value=5.0)
        self.contour_max_depth = tk.DoubleVar(value=40.0)
        self.contour_step = tk.DoubleVar(value=1.0)
        self.fig_width = tk.DoubleVar(value=8.0)
        self.fig_dpi = tk.IntVar(value=100)
        self.tile_opacity = tk.DoubleVar(value=0.9)
        self.contour_opacity = tk.DoubleVar(value=0.9)

        # Optimization parameters
        self.n_workers = tk.IntVar(value=0)  # 0 = auto-detect
        self.use_parallel = tk.BooleanVar(value=True)
        self.enable_contours = tk.BooleanVar(value=True)
        self.show_depth_labels = tk.BooleanVar(value=True)

        # Optional bounds
        self.use_custom_bounds = tk.BooleanVar()
        self.min_lat = tk.DoubleVar()
        self.max_lat = tk.DoubleVar()
        self.min_lon = tk.DoubleVar()
        self.max_lon = tk.DoubleVar()

        self.is_generating = False
        self.stop_generation = False

        # Store validated entry widgets for validation checking
        self.validated_entries = []

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # CSV File selection with validation
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W)
        csv_entry_frame = ttk.Frame(file_frame)
        csv_entry_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 5))

        self.csv_entry = ValidatedEntry(
            csv_entry_frame,
            self.csv_file,
            validator_func=lambda x: InputValidator.validate_file_path(x, must_exist=True, extension='.csv'),
            width=40
        )
        self.csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.validated_entries.append(self.csv_entry)

        ttk.Button(file_frame, text="Browse", command=self.browse_csv).grid(row=0, column=2)

        # Output Directory selection with validation
        ttk.Label(file_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        output_entry_frame = ttk.Frame(file_frame)
        output_entry_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=(5, 0))

        self.output_entry = ValidatedEntry(
            output_entry_frame,
            self.output_dir,
            validator_func=lambda x: InputValidator.validate_directory_path(x, must_exist=False, create_if_missing=False),
            width=40
        )
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.validated_entries.append(self.output_entry)

        ttk.Button(file_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=(5, 0))

        # Configure file_frame column weights
        file_frame.columnconfigure(1, weight=1)

        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # Left column
        left_config = ttk.Frame(config_frame)
        left_config.grid(row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 20))

        # Tile Size validation
        ttk.Label(left_config, text="Tile Size (degrees):").grid(row=0, column=0, sticky=tk.W)
        tile_size_entry = ValidatedEntry(
            left_config,
            self.tile_size,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=0.001, max_val=10.0),
            width=15
        )
        tile_size_entry.grid(row=0, column=1, padx=(10, 0))
        self.validated_entries.append(tile_size_entry)

        # Overlap Factor validation
        ttk.Label(left_config, text="Overlap Factor:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        overlap_entry = ValidatedEntry(
            left_config,
            self.overlap_factor,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=0.0, max_val=1.0),
            width=15
        )
        overlap_entry.grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(overlap_entry)

        # Resolution validation
        ttk.Label(left_config, text="Resolution:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        resolution_entry = ValidatedEntry(
            left_config,
            self.resolution,
            validator_func=lambda x: InputValidator.validate_int(x, min_val=100, max_val=10000),
            width=15
        )
        resolution_entry.grid(row=2, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(resolution_entry)

        # Batch Size validation
        ttk.Label(left_config, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        batch_entry = ValidatedEntry(
            left_config,
            self.batch_size,
            validator_func=lambda x: InputValidator.validate_int(x, min_val=1, max_val=100),
            width=15
        )
        batch_entry.grid(row=3, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(batch_entry)

        # Smoothing Sigma validation
        ttk.Label(left_config, text="Smoothing Sigma:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        sigma_entry = ValidatedEntry(
            left_config,
            self.smoothing_sigma,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=0.1, max_val=50.0),
            width=15
        )
        sigma_entry.grid(row=4, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(sigma_entry)

        # Right column
        right_config = ttk.Frame(config_frame)
        right_config.grid(row=0, column=1, sticky=(tk.W, tk.N), padx=(20, 0))

        # Max Depth validation
        ttk.Label(right_config, text="Max Depth:").grid(row=0, column=0, sticky=tk.W)
        max_depth_entry = ValidatedEntry(
            right_config,
            self.contour_max_depth,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=1.0, max_val=1000.0),
            width=15
        )
        max_depth_entry.grid(row=0, column=1, padx=(10, 0))
        self.validated_entries.append(max_depth_entry)

        # Contour Step validation
        ttk.Label(right_config, text="Contour Step:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        step_entry = ValidatedEntry(
            right_config,
            self.contour_step,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=0.1, max_val=10.0),
            width=15
        )
        step_entry.grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(step_entry)

        # Figure Width validation
        ttk.Label(right_config, text="Figure Width:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        width_entry = ValidatedEntry(
            right_config,
            self.fig_width,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=1.0, max_val=20.0),
            width=15
        )
        width_entry.grid(row=2, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(width_entry)

        # Figure DPI validation
        ttk.Label(right_config, text="Figure DPI:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        dpi_entry = ValidatedEntry(
            right_config,
            self.fig_dpi,
            validator_func=lambda x: InputValidator.validate_int(x, min_val=50, max_val=500),
            width=15
        )
        dpi_entry.grid(row=3, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(dpi_entry)

        # Tile Opacity validation
        ttk.Label(right_config, text="Tile Opacity:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        tile_opacity_entry = ValidatedEntry(
            right_config,
            self.tile_opacity,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=0.0, max_val=1.0),
            width=15
        )
        tile_opacity_entry.grid(row=4, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(tile_opacity_entry)

        # Contour Opacity validation
        ttk.Label(right_config, text="Contour Opacity:").grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
        contour_opacity_entry = ValidatedEntry(
            right_config,
            self.contour_opacity,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=0.0, max_val=1.0),
            width=15
        )
        contour_opacity_entry.grid(row=5, column=1, padx=(10, 0), pady=(5, 0))
        self.validated_entries.append(contour_opacity_entry)

        # Contour options in right column
        ttk.Checkbutton(right_config, text="Enable contour lines",
                       variable=self.enable_contours).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        ttk.Checkbutton(right_config, text="Show depth labels",
                       variable=self.show_depth_labels).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # Optimization section
        optimization_frame = ttk.LabelFrame(main_frame, text="Performance Optimization", padding="10")
        optimization_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N), padx=(10, 0), pady=(0, 10))

        # Workers validation
        ttk.Label(optimization_frame, text="Workers (0=auto):").grid(row=0, column=0, sticky=tk.W)
        workers_entry = ValidatedEntry(
            optimization_frame,
            self.n_workers,
            validator_func=lambda x: InputValidator.validate_int(x, min_val=0, max_val=16),
            width=10
        )
        workers_entry.grid(row=0, column=1, padx=(10, 0))
        self.validated_entries.append(workers_entry)

        ttk.Checkbutton(optimization_frame, text="Use parallel processing",
                       variable=self.use_parallel).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # Custom bounds section
        bounds_frame = ttk.LabelFrame(main_frame, text="Custom Bounds (Optional)", padding="10")
        bounds_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Checkbutton(bounds_frame, text="Use custom bounds", variable=self.use_custom_bounds,
                       command=self.toggle_bounds).grid(row=0, column=0, columnspan=4, sticky=tk.W)

        self.bounds_widgets = []

        # Min Latitude validation
        ttk.Label(bounds_frame, text="Min Latitude:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.min_lat_entry = ValidatedEntry(
            bounds_frame,
            self.min_lat,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=-90.0, max_val=90.0, allow_empty=True),
            width=15,
            state='disabled'
        )
        self.min_lat_entry.grid(row=1, column=1, padx=(10, 20), pady=(5, 0))
        self.bounds_widgets.append(self.min_lat_entry)
        self.validated_entries.append(self.min_lat_entry)

        # Max Latitude validation
        ttk.Label(bounds_frame, text="Max Latitude:").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))
        self.max_lat_entry = ValidatedEntry(
            bounds_frame,
            self.max_lat,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=-90.0, max_val=90.0, allow_empty=True),
            width=15,
            state='disabled'
        )
        self.max_lat_entry.grid(row=1, column=3, padx=(10, 0), pady=(5, 0))
        self.bounds_widgets.append(self.max_lat_entry)
        self.validated_entries.append(self.max_lat_entry)

        # Min Longitude validation
        ttk.Label(bounds_frame, text="Min Longitude:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.min_lon_entry = ValidatedEntry(
            bounds_frame,
            self.min_lon,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=-180.0, max_val=180.0, allow_empty=True),
            width=15,
            state='disabled'
        )
        self.min_lon_entry.grid(row=2, column=1, padx=(10, 20), pady=(5, 0))
        self.bounds_widgets.append(self.min_lon_entry)
        self.validated_entries.append(self.min_lon_entry)

        # Max Longitude validation
        ttk.Label(bounds_frame, text="Max Longitude:").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))
        self.max_lon_entry = ValidatedEntry(
            bounds_frame,
            self.max_lon,
            validator_func=lambda x: InputValidator.validate_float(x, min_val=-180.0, max_val=180.0, allow_empty=True),
            width=15,
            state='disabled'
        )
        self.max_lon_entry.grid(row=2, column=3, padx=(10, 0), pady=(5, 0))
        self.bounds_widgets.append(self.max_lon_entry)
        self.validated_entries.append(self.max_lon_entry)

        # Control section
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        self.generate_btn = ttk.Button(control_frame, text="Generate Tiles", command=self.start_generation)
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_generation_process, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(control_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT)

        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state='disabled')
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure grid weights for responsive layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=0)
        main_frame.rowconfigure(5, weight=1)

    def browse_csv(self):
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_file.set(filename)

    def browse_output(self):
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_dir.set(directory)

    def toggle_bounds(self):
        state = 'normal' if self.use_custom_bounds.get() else 'disabled'
        for widget in self.bounds_widgets:
            widget.entry.configure(state=state)

    def log_message(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        self.root.update_idletasks()

    def progress_callback(self, current, total, message):
        """Callback function to update progress from tile generator."""
        def update_progress():
            percentage = (current / total) * 100 if total > 0 else 0
            self.progress_bar['value'] = percentage
            self.progress_var.set(f"{message} ({percentage:.1f}%)")
            self.root.update_idletasks()

        # Schedule the update on the main thread
        self.root.after(0, update_progress)

    def should_stop_generation(self):
        """Check if generation should be stopped."""
        return self.stop_generation

    def stop_generation_process(self):
        """Stop the tile generation process."""
        if self.is_generating:
            self.stop_generation = True
            self.log_message("Stopping generation...")
            self.progress_var.set("Stopping generation...")
            self.stop_btn.configure(state='disabled')

    def create_config(self):
        n_workers = self.n_workers.get() if self.n_workers.get() > 0 else None
        return TilerConfig(
            tile_size_deg=self.tile_size.get(),
            overlap_factor=self.overlap_factor.get(),
            resolution=self.resolution.get(),
            batch_size=self.batch_size.get(),
            smoothing_sigma=self.smoothing_sigma.get(),
            contour_max_depth=self.contour_max_depth.get(),
            contour_step=self.contour_step.get(),
            fig_width=self.fig_width.get(),
            fig_dpi=self.fig_dpi.get(),
            tile_opacity=self.tile_opacity.get(),
            contour_opacity=self.contour_opacity.get(),
            n_workers=n_workers,
            enable_contours=self.enable_contours.get(),
            show_depth_labels=self.show_depth_labels.get()
        )

    def save_config(self):
        # Validate inputs before saving
        validation_errors = self.validate_all_inputs()

        # Only show dialog if there are validation errors
        if validation_errors:
            dialog = ValidationDialog(
                self.root,
                "Save Configuration - Validation",
                validation_errors,
                allow_proceed=True
            )

            if not dialog.result:
                return  # User chose to fix issues first

        filename = filedialog.asksaveasfilename(
            title="Save configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            config_data = {
                'tile_size_deg': self.tile_size.get(),
                'overlap_factor': self.overlap_factor.get(),
                'resolution': self.resolution.get(),
                'batch_size': self.batch_size.get(),
                'smoothing_sigma': self.smoothing_sigma.get(),
                'contour_max_depth': self.contour_max_depth.get(),
                'contour_step': self.contour_step.get(),
                'fig_width': self.fig_width.get(),
                'fig_dpi': self.fig_dpi.get(),
                'tile_opacity': self.tile_opacity.get(),
                'contour_opacity': self.contour_opacity.get(),
                'n_workers': self.n_workers.get(),
                'use_parallel': self.use_parallel.get(),
                'enable_contours': self.enable_contours.get(),
                'show_depth_labels': self.show_depth_labels.get(),
                'use_custom_bounds': self.use_custom_bounds.get(),
                'min_lat': self.min_lat.get() if self.use_custom_bounds.get() else None,
                'max_lat': self.max_lat.get() if self.use_custom_bounds.get() else None,
                'min_lon': self.min_lon.get() if self.use_custom_bounds.get() else None,
                'max_lon': self.max_lon.get() if self.use_custom_bounds.get() else None,
                'csv_file': self.csv_file.get(),
                'output_dir': self.output_dir.get()
            }

            try:
                with open(filename, 'w') as f:
                    json.dump(config_data, f, indent=2)
                self.log_message(f"Configuration saved to {filename}")
                if validation_errors:
                    self.log_message("Note: Configuration saved with validation warnings")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        filename = filedialog.askopenfilename(
            title="Load configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_data = json.load(f)

                # Update variables
                self.tile_size.set(config_data.get('tile_size_deg', 0.01))
                self.overlap_factor.set(config_data.get('overlap_factor', 0.1))
                self.resolution.set(config_data.get('resolution', 1000))
                self.batch_size.set(config_data.get('batch_size', 10))
                self.smoothing_sigma.set(config_data.get('smoothing_sigma', 5.0))
                self.contour_max_depth.set(config_data.get('contour_max_depth', 40.0))
                self.contour_step.set(config_data.get('contour_step', 1.0))
                self.fig_width.set(config_data.get('fig_width', 8.0))
                self.fig_dpi.set(config_data.get('fig_dpi', 100))
                self.tile_opacity.set(config_data.get('tile_opacity', 0.9))
                self.contour_opacity.set(config_data.get('contour_opacity', 0.9))
                self.n_workers.set(config_data.get('n_workers', 0))
                self.use_parallel.set(config_data.get('use_parallel', True))
                self.enable_contours.set(config_data.get('enable_contours', True))
                self.show_depth_labels.set(config_data.get('show_depth_labels', True))

                self.use_custom_bounds.set(config_data.get('use_custom_bounds', False))
                if config_data.get('min_lat') is not None:
                    self.min_lat.set(config_data['min_lat'])
                if config_data.get('max_lat') is not None:
                    self.max_lat.set(config_data['max_lat'])
                if config_data.get('min_lon') is not None:
                    self.min_lon.set(config_data['min_lon'])
                if config_data.get('max_lon') is not None:
                    self.max_lon.set(config_data['max_lon'])

                self.csv_file.set(config_data.get('csv_file', ''))
                self.output_dir.set(config_data.get('output_dir', 'tiles'))

                self.toggle_bounds()
                self.log_message(f"Configuration loaded from {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def validate_all_inputs(self):
        """Validate all inputs and collect detailed error messages"""
        validation_errors = []

        # Field name mapping for better error messages
        field_names = {
            id(self.csv_entry): "CSV File",
            id(self.output_entry): "Output Directory",
            id(self.validated_entries[2]): "Tile Size",
            id(self.validated_entries[3]): "Overlap Factor",
            id(self.validated_entries[4]): "Resolution",
            id(self.validated_entries[5]): "Batch Size",
            id(self.validated_entries[6]): "Smoothing Sigma",
            id(self.validated_entries[7]): "Max Depth",
            id(self.validated_entries[8]): "Contour Step",
            id(self.validated_entries[9]): "Figure Width",
            id(self.validated_entries[10]): "Figure DPI",
            id(self.validated_entries[11]): "Tile Opacity",
            id(self.validated_entries[12]): "Contour Opacity",
            id(self.validated_entries[13]): "Workers",
            id(self.min_lat_entry): "Min Latitude",
            id(self.max_lat_entry): "Max Latitude",
            id(self.min_lon_entry): "Min Longitude",
            id(self.max_lon_entry): "Max Longitude"
        }

        # Force validation on all validated entries and collect specific errors
        for entry in self.validated_entries:
            if not entry.force_validate():
                field_name = field_names.get(id(entry), "Unknown field")
                error_msg = entry.get_error_message()
                validation_errors.append(f"{field_name}: {error_msg}")

        # Additional logical validations
        if self.use_custom_bounds.get():
            try:
                min_lat = self.min_lat.get()
                max_lat = self.max_lat.get()
                min_lon = self.min_lon.get()
                max_lon = self.max_lon.get()

                if min_lat >= max_lat:
                    validation_errors.append("Coordinates: Min latitude must be less than max latitude")
                if min_lon >= max_lon:
                    validation_errors.append("Coordinates: Min longitude must be less than max longitude")
            except tk.TclError:
                validation_errors.append("Coordinates: Invalid coordinate values")

        # Validate contour step vs max depth
        try:
            if self.contour_step.get() > self.contour_max_depth.get():
                validation_errors.append("Contour Settings: Contour step cannot be larger than max depth")
        except tk.TclError:
            pass

        return validation_errors

    def start_generation(self):
        if self.is_generating:
            return

        # Comprehensive validation
        validation_errors = self.validate_all_inputs()

        # Only show dialog if there are validation errors
        if validation_errors:
            dialog = ValidationDialog(
                self.root,
                "Generate Tiles - Validation",
                validation_errors,
                allow_proceed=False  # Don't allow proceeding with errors for generation
            )

            if not dialog.result:
                return  # User chose to fix issues first

        # Additional file checks
        csv_path = self.csv_file.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select a CSV file")
            return

        if not Path(csv_path).exists():
            messagebox.showerror("Error", "CSV file does not exist")
            return

        # Validate output directory
        output_path = self.output_dir.get()
        if not output_path:
            messagebox.showerror("Error", "Please specify an output directory")
            return

        # Try to create output directory if it doesn't exist
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create output directory: {e}")
            return

        self.is_generating = True
        self.stop_generation = False
        self.generate_btn.configure(text="Generating...", state='disabled')
        self.stop_btn.configure(state='normal')
        self.progress_bar['value'] = 0
        self.progress_var.set("Starting generation...")
        self.log_message("All validations passed. Starting generation...")

        # Run generation in separate thread
        thread = threading.Thread(target=self.generate_tiles)
        thread.daemon = True
        thread.start()

    def generate_tiles(self):
        try:
            config = self.create_config()

            # Get custom bounds if specified
            bounds = {}
            if self.use_custom_bounds.get():
                bounds = {
                    'min_lat': self.min_lat.get(),
                    'max_lat': self.max_lat.get(),
                    'min_lon': self.min_lon.get(),
                    'max_lon': self.max_lon.get()
                }

            self.log_message("Creating tiler...")
            self.log_message(f"CSV file: {self.csv_file.get()}")
            self.log_message(f"Output directory: {self.output_dir.get()}")

            if bounds:
                self.log_message(f"Using custom bounds: {bounds}")

            tiler = BathymetryTiler(
                csv_file=self.csv_file.get(),
                output_dir=self.output_dir.get(),
                config=config,
                **bounds
            )

            self.log_message("Starting tile generation...")

            metadata = tiler.generate_all_tiles(
                progress_callback=self.progress_callback,
                should_stop_callback=self.should_stop_generation,
                use_parallel=self.use_parallel.get()
            )

            if self.stop_generation:
                self.log_message(f"Generation stopped by user - Generated {len(metadata)} tiles")
                self.progress_var.set(f"Stopped - Generated {len(metadata)} tiles")
            else:
                self.log_message(f"Successfully generated {len(metadata)} tiles")
                self.progress_var.set(f"Completed - Generated {len(metadata)} tiles")
                messagebox.showinfo("Success", f"Successfully generated {len(metadata)} tiles!")

        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            self.log_message(f"Error: {error_msg}")
            self.progress_var.set("Error: File not found")
            messagebox.showerror("File Error", error_msg)

        except PermissionError as e:
            error_msg = f"Permission denied: {str(e)}"
            self.log_message(f"Error: {error_msg}")
            self.progress_var.set("Error: Permission denied")
            messagebox.showerror("Permission Error", error_msg)

        except ValueError as e:
            error_msg = f"Invalid data or configuration: {str(e)}"
            self.log_message(f"Error: {error_msg}")
            self.progress_var.set("Error: Invalid data")
            messagebox.showerror("Data Error", error_msg)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.log_message(f"Error: {error_msg}")
            self.progress_var.set("Error occurred")
            messagebox.showerror("Error", f"Generation failed: {error_msg}")

        finally:
            self.is_generating = False
            self.stop_generation = False
            self.generate_btn.configure(text="Generate Tiles", state='normal')
            self.stop_btn.configure(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = TileGeneratorGUI(root)
    root.mainloop()