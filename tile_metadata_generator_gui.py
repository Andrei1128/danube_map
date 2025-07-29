import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import json
import glob
from pathlib import Path
import re


class TileMetadataGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tile Metadata Generator")
        self.root.geometry("700x500")

        self.tiles_dir = tk.StringVar(value="bathymetry_tiles")
        self.output_file = tk.StringVar(value="tiles_metadata.json")

        self.is_generating = False
        self.stop_generation = False
        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(file_frame, text="Tiles Directory:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.tiles_dir, width=50).grid(row=0, column=1, padx=(10, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_tiles_dir).grid(row=0, column=2)

        ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(file_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, padx=(10, 5), pady=(5, 0))
        ttk.Button(file_frame, text="Browse", command=self.browse_output_file).grid(row=1, column=2, pady=(5, 0))

        # Control section
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.generate_btn = ttk.Button(control_frame, text="Generate Metadata", command=self.start_generation)
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_generation_process, state='disabled')
        self.stop_btn.pack(side=tk.LEFT)

        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD, state='disabled')
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

    def browse_tiles_dir(self):
        directory = filedialog.askdirectory(
            title="Select tiles directory",
            initialdir=self.tiles_dir.get()
        )
        if directory:
            self.tiles_dir.set(directory)

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Save metadata file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=self.output_file.get()
        )
        if filename:
            self.output_file.set(filename)

    def log_message(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        self.root.update_idletasks()

    def progress_callback(self, current, total, message):
        """Callback function to update progress."""
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
        """Stop the metadata generation process."""
        if self.is_generating:
            self.stop_generation = True
            self.log_message("Stopping generation...")
            self.progress_var.set("Stopping generation...")
            self.stop_btn.configure(state='disabled')

    def _get_output_path(self):
        """Get the resolved output file path."""
        output_file = self.output_file.get()
        if os.path.isabs(output_file):
            return Path(output_file)
        else:
            return Path(self.tiles_dir.get()) / output_file

    def _save_metadata(self, coordinates_data):
        """Save metadata to file."""
        output_path = self._get_output_path()
        with open(output_path, 'w') as f:
            json.dump(coordinates_data, f, indent=2)
        return output_path

    def extract_coordinates_from_filename(self, filename):
        """
        Extract coordinates from SVG filename in format:
        tile_{min_lat}_{min_lon}_{max_lat}_{max_lon}.svg
        """
        pattern = r'tile_([0-9.]+)_([0-9.]+)_([0-9.]+)_([0-9.]+)\.svg'
        match = re.match(pattern, filename)

        if match:
            min_lat = float(match.group(1))
            min_lon = float(match.group(2))
            max_lat = float(match.group(3))
            max_lon = float(match.group(4))

            return {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon
            }
        return None

    def start_generation(self):
        if self.is_generating:
            return

        if not self.tiles_dir.get():
            messagebox.showerror("Error", "Please select a tiles directory")
            return

        if not Path(self.tiles_dir.get()).exists():
            messagebox.showerror("Error", "Tiles directory does not exist")
            return

        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file")
            return

        self.is_generating = True
        self.stop_generation = False
        self.generate_btn.configure(text="Generating...", state='disabled')
        self.stop_btn.configure(state='normal')
        self.progress_bar['value'] = 0
        self.progress_var.set("Starting metadata generation...")

        # Run generation in separate thread
        thread = threading.Thread(target=self.generate_metadata)
        thread.daemon = True
        thread.start()

    def generate_metadata(self):
        try:
            tiles_dir = self.tiles_dir.get()

            self.log_message(f"Scanning directory: {tiles_dir}")

            # Find all SVG files
            files = glob.glob(os.path.join(tiles_dir, "*.svg"))

            if not files:
                self.log_message(f"No SVG files found in '{tiles_dir}'")
                self.progress_var.set("No SVG files found")
                messagebox.showwarning("Warning", f"No SVG files found in '{tiles_dir}'")
                return

            total_files = len(files)
            self.log_message(f"Found {total_files} SVG files")

            coordinates_data = []
            processed = 0

            self.progress_callback(0, total_files, "Starting file processing...")

            for path in files:
                # Check if we should stop
                if self.should_stop_generation():
                    self.log_message(f"Processing stopped after {processed} files")
                    break

                filename = os.path.basename(path)
                coordinates = self.extract_coordinates_from_filename(filename)

                if coordinates:
                    coordinates_data.append({
                        "filename": filename,
                        "coordinates": coordinates
                    })
                else:
                    self.log_message(f"Warning: Could not extract coordinates from: {filename}")

                processed += 1
                self.progress_callback(processed, total_files, f"Processed {processed}/{total_files} files")

            # Save metadata if we have data
            if coordinates_data:
                output_path = self._save_metadata(coordinates_data)
                
                if not self.stop_generation:
                    self.log_message(f"Successfully extracted coordinates from {len(coordinates_data)} SVG files")
                    self.log_message(f"Output saved to: {output_path}")
                    self.progress_var.set(f"Completed - Generated metadata for {len(coordinates_data)} files")
                else:
                    self.log_message(f"Generation stopped - Saved metadata for {len(coordinates_data)} files")
                    self.progress_var.set(f"Stopped - Generated metadata for {len(coordinates_data)} files")
            elif self.stop_generation:
                self.log_message("Generation stopped - No files processed")
                self.progress_var.set("Stopped - No files processed")

        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.progress_var.set("Error occurred")
            messagebox.showerror("Error", f"Generation failed: {str(e)}")

        finally:
            self.is_generating = False
            self.stop_generation = False
            self.generate_btn.configure(text="Generate Metadata", state='normal')
            self.stop_btn.configure(state='disabled')


if __name__ == "__main__":
    root = tk.Tk()
    app = TileMetadataGeneratorGUI(root)
    root.mainloop()