import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import json
from tile_generator import BathymetryTiler, TilerConfig


class TileGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bathymetry Tile Generator")
        self.root.geometry("800x600")
        
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
        
        # Optional bounds
        self.use_custom_bounds = tk.BooleanVar()
        self.min_lat = tk.DoubleVar()
        self.max_lat = tk.DoubleVar()
        self.min_lon = tk.DoubleVar()
        self.max_lon = tk.DoubleVar()
        
        self.is_generating = False
        self.stop_generation = False
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.csv_file, width=50).grid(row=0, column=1, padx=(10, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_csv).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(file_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=(10, 5), pady=(5, 0))
        ttk.Button(file_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=(5, 0))
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Left column
        left_config = ttk.Frame(config_frame)
        left_config.grid(row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 20))
        
        ttk.Label(left_config, text="Tile Size (degrees):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(left_config, textvariable=self.tile_size, width=15).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(left_config, text="Overlap Factor:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(left_config, textvariable=self.overlap_factor, width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(left_config, text="Resolution:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(left_config, textvariable=self.resolution, width=15).grid(row=2, column=1, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(left_config, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(left_config, textvariable=self.batch_size, width=15).grid(row=3, column=1, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(left_config, text="Smoothing Sigma:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(left_config, textvariable=self.smoothing_sigma, width=15).grid(row=4, column=1, padx=(10, 0), pady=(5, 0))
        
        # Right column
        right_config = ttk.Frame(config_frame)
        right_config.grid(row=0, column=1, sticky=(tk.W, tk.N))
        
        ttk.Label(right_config, text="Max Depth:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(right_config, textvariable=self.contour_max_depth, width=15).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(right_config, text="Contour Step:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(right_config, textvariable=self.contour_step, width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(right_config, text="Figure Width:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(right_config, textvariable=self.fig_width, width=15).grid(row=2, column=1, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(right_config, text="Figure DPI:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(right_config, textvariable=self.fig_dpi, width=15).grid(row=3, column=1, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(right_config, text="Tile Opacity:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(right_config, textvariable=self.tile_opacity, width=15).grid(row=4, column=1, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(right_config, text="Contour Opacity:").grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(right_config, textvariable=self.contour_opacity, width=15).grid(row=5, column=1, padx=(10, 0), pady=(5, 0))
        
        # Custom bounds section
        bounds_frame = ttk.LabelFrame(main_frame, text="Custom Bounds (Optional)", padding="10")
        bounds_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(bounds_frame, text="Use custom bounds", variable=self.use_custom_bounds, 
                       command=self.toggle_bounds).grid(row=0, column=0, columnspan=4, sticky=tk.W)
        
        self.bounds_widgets = []
        
        ttk.Label(bounds_frame, text="Min Latitude:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        min_lat_entry = ttk.Entry(bounds_frame, textvariable=self.min_lat, width=15, state='disabled')
        min_lat_entry.grid(row=1, column=1, padx=(10, 20), pady=(5, 0))
        self.bounds_widgets.append(min_lat_entry)
        
        ttk.Label(bounds_frame, text="Max Latitude:").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))
        max_lat_entry = ttk.Entry(bounds_frame, textvariable=self.max_lat, width=15, state='disabled')
        max_lat_entry.grid(row=1, column=3, padx=(10, 0), pady=(5, 0))
        self.bounds_widgets.append(max_lat_entry)
        
        ttk.Label(bounds_frame, text="Min Longitude:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        min_lon_entry = ttk.Entry(bounds_frame, textvariable=self.min_lon, width=15, state='disabled')
        min_lon_entry.grid(row=2, column=1, padx=(10, 20), pady=(5, 0))
        self.bounds_widgets.append(min_lon_entry)
        
        ttk.Label(bounds_frame, text="Max Longitude:").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))
        max_lon_entry = ttk.Entry(bounds_frame, textvariable=self.max_lon, width=15, state='disabled')
        max_lon_entry.grid(row=2, column=3, padx=(10, 0), pady=(5, 0))
        self.bounds_widgets.append(max_lon_entry)
        
        # Control section
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.generate_btn = ttk.Button(control_frame, text="Generate Tiles", command=self.start_generation)
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_generation_process, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
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
            widget.configure(state=state)
    
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
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
            contour_opacity=self.contour_opacity.get()
        )
    
    def save_config(self):
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
    
    def start_generation(self):
        if self.is_generating:
            return
        
        if not self.csv_file.get():
            messagebox.showerror("Error", "Please select a CSV file")
            return
        
        if not Path(self.csv_file.get()).exists():
            messagebox.showerror("Error", "CSV file does not exist")
            return
        
        self.is_generating = True
        self.stop_generation = False
        self.generate_btn.configure(text="Generating...", state='disabled')
        self.stop_btn.configure(state='normal')
        self.progress_bar['value'] = 0
        self.progress_var.set("Starting generation...")
        
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
            tiler = BathymetryTiler(
                csv_file=self.csv_file.get(),
                output_dir=self.output_dir.get(),
                config=config,
                **bounds
            )
            
            self.log_message("Starting tile generation...")
            
            metadata = tiler.generate_all_tiles(
                progress_callback=self.progress_callback,
                should_stop_callback=self.should_stop_generation
            )
            
            if self.stop_generation:
                self.log_message(f"Generation stopped - Generated {len(metadata)} tiles")
                self.progress_var.set(f"Stopped - Generated {len(metadata)} tiles")
            else:
                self.log_message(f"Successfully generated {len(metadata)} tiles")
                self.progress_var.set(f"Completed - Generated {len(metadata)} tiles")
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.progress_var.set("Error occurred")
            messagebox.showerror("Error", f"Generation failed: {str(e)}")
        
        finally:
            self.is_generating = False
            self.stop_generation = False
            self.generate_btn.configure(text="Generate Tiles", state='normal')
            self.stop_btn.configure(state='disabled')


def main():
    root = tk.Tk()
    app = TileGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()