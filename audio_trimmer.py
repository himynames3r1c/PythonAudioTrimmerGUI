import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioTrimmer:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Trimmer")
        self.root.geometry("1000x800")  # Increased height for better spacing

        self.audio = None
        self.sample_rate = 44100
        self.audio_duration = 0
        self.time_unit = "seconds"
        self.dragging = None  # Tracks if user is dragging start or end line
        self.updating_sliders = False

        # File Selection
        tk.Label(root, text="Select Audio File:").pack(pady=5)
        self.entry_file = tk.Entry(root, width=50)
        self.entry_file.pack(padx=10)
        tk.Button(root, text="Browse", command=self.select_file).pack(pady=5)

        # Time Unit Selection (Seconds/Milliseconds)
        self.unit_var = tk.StringVar(value="seconds")
        unit_frame = tk.Frame(root)
        tk.Label(unit_frame, text="Time Unit:").pack(side=tk.LEFT)
        tk.Radiobutton(unit_frame, text="Seconds", variable=self.unit_var, value="seconds", command=self.update_unit).pack(side=tk.LEFT)
        tk.Radiobutton(unit_frame, text="Milliseconds", variable=self.unit_var, value="milliseconds", command=self.update_unit).pack(side=tk.LEFT)
        unit_frame.pack(pady=5)

        # Start & End Time Sliders
        self.start_var = tk.DoubleVar()
        self.end_var = tk.DoubleVar()

        tk.Label(root, text="Select Start Time:").pack()
        self.start_slider = tk.Scale(root, from_=0, to=100, orient="horizontal", length=500, variable=self.start_var, resolution=0.001, command=self.slider_moved)
        self.start_slider.pack()

        tk.Label(root, text="Select End Time:").pack()
        self.end_slider = tk.Scale(root, from_=0, to=100, orient="horizontal", length=500, variable=self.end_var, resolution=0.001, command=self.slider_moved)
        self.end_slider.pack()

        # Trim Button
        tk.Button(root, text="Trim Audio", command=self.trim_audio, bg="green", fg="white").pack(pady=10)

        # Adjusted Figure Size & Spacing
        self.figure, (self.ax_waveform, self.ax_spectrum) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})  # More space for waveform
        self.figure.subplots_adjust(hspace=0.4)  # Increase vertical spacing between plots

        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        # Connect waveform click & drag events
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def select_file(self):
        """Open file dialog to select an audio file."""
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav;*.ogg;*.flac")])
        if file_path:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, file_path)
            self.load_audio(file_path)

    def load_audio(self, file_path):
        """Load the selected audio file and update sliders."""
        self.audio = AudioSegment.from_file(file_path)
        self.sample_rate = self.audio.frame_rate
        self.audio_duration = len(self.audio) / 1000  # Convert to seconds

        self.update_unit()

    def update_unit(self):
        """Update slider scale based on seconds or milliseconds."""
        unit = self.unit_var.get()
        max_value = self.audio_duration if unit == "seconds" else self.audio_duration * 1000
        resolution = 0.001 if unit == "seconds" else 1  # Millisecond precision

        self.start_slider.config(from_=0, to=max_value, resolution=resolution)
        self.end_slider.config(from_=0, to=max_value, resolution=resolution)

        self.start_var.set(0)
        self.end_var.set(max_value)

        self.update_waveform(None)

    def update_waveform(self, event):
        """Update waveform visualization."""
        if self.audio is None:
            return

        start_ms = int(self.start_var.get() * 1000)
        end_ms = int(self.end_var.get() * 1000)

        trimmed_audio = self.audio[start_ms:end_ms]

        # Convert to numpy & normalize
        samples = np.array(trimmed_audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.max(np.abs(samples)) if len(samples) > 0 else samples

        time_axis = np.linspace(self.start_var.get(), self.end_var.get(), num=len(samples))

        self.ax_waveform.clear()
        self.ax_waveform.plot(time_axis, samples, color="blue")
        self.ax_waveform.axvline(self.start_var.get(), color="red", linestyle="--", label="Start")
        self.ax_waveform.axvline(self.end_var.get(), color="green", linestyle="--", label="End")
        self.ax_waveform.set_title("Trimmed Audio Waveform")
        self.ax_waveform.legend()

        # Frequency spectrum
        if len(samples) > 0:
            fft_data = np.abs(fft(samples))[:len(samples) // 2]
            freqs = np.fft.fftfreq(len(samples), d=1 / self.sample_rate)[:len(samples) // 2]

            self.ax_spectrum.clear()
            self.ax_spectrum.plot(freqs, fft_data, color="red")
            self.ax_spectrum.set_title("Frequency Spectrum")

        self.canvas.draw()

    def slider_moved(self, event):
        """Handles slider changes while preventing infinite updates."""
        if self.updating_sliders:
            return
        self.updating_sliders = True
        self.update_waveform(None)
        self.updating_sliders = False

    def on_press(self, event):
        """Start dragging selection on waveform."""
        if self.audio is None or event.xdata is None:
            return
        if abs(event.xdata - self.start_var.get()) < abs(event.xdata - self.end_var.get()):
            self.dragging = "start"
            self.start_var.set(event.xdata)
        else:
            self.dragging = "end"
            self.end_var.set(event.xdata)

    def on_drag(self, event):
        """Drag selection across waveform."""
        if self.dragging and event.xdata:
            if self.dragging == "start":
                self.start_var.set(event.xdata)
            elif self.dragging == "end":
                self.end_var.set(event.xdata)
            self.update_waveform(None)

    def on_release(self, event):
        """Release drag selection on waveform."""
        self.dragging = None

    def trim_audio(self):
        """Trim the audio based on selection and save."""
        if self.audio is None:
            messagebox.showerror("Error", "No audio file selected.")
            return

        start_ms = int(self.start_var.get() * 1000)
        end_ms = int(self.end_var.get() * 1000)

        if start_ms >= end_ms:
            messagebox.showerror("Error", "End time must be greater than start time.")
            return

        trimmed_audio = self.audio[start_ms:end_ms]

        output_path = filedialog.asksaveasfilename(defaultextension=".mp3",
            filetypes=[("MP3 Files", "*.mp3"), ("WAV Files", "*.wav"), ("OGG Files", "*.ogg")])
        if output_path:
            trimmed_audio.export(output_path, format=output_path.split(".")[-1])
            messagebox.showinfo("Success", f"Trimmed audio saved at:\n{output_path}")

root = tk.Tk()
app = AudioTrimmer(root)
root.mainloop()
