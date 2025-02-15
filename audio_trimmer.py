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
        self.root.geometry("900x750")

        self.audio = None
        self.sample_rate = 44100
        self.audio_duration = 0
        self.time_unit = "seconds"  # Default unit

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
        self.start_slider = tk.Scale(root, from_=0, to=100, orient="horizontal", length=500, variable=self.start_var, command=self.update_waveform, resolution=0.001)
        self.start_slider.pack()

        tk.Label(root, text="Select End Time:").pack()
        self.end_slider = tk.Scale(root, from_=0, to=100, orient="horizontal", length=500, variable=self.end_var, command=self.update_waveform, resolution=0.001)
        self.end_slider.pack()

        # Trim Button
        tk.Button(root, text="Trim Audio", command=self.trim_audio, bg="green", fg="white").pack(pady=10)

        # Waveform and Frequency Spectrum Display
        self.figure, (self.ax_waveform, self.ax_spectrum) = plt.subplots(2, 1, figsize=(8, 5))

        self.ax_waveform.set_title("Audio Waveform")
        self.ax_waveform.set_xlabel("Time (seconds)")
        self.ax_waveform.set_ylabel("Amplitude")

        self.ax_spectrum.set_title("Frequency Spectrum")
        self.ax_spectrum.set_xlabel("Frequency (Hz)")
        self.ax_spectrum.set_ylabel("Magnitude")

        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

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

        # Update slider range
        self.update_unit()

        # Display waveform and frequency spectrum
        self.update_waveform(None)

    def update_unit(self):
        """Update the slider scale based on the selected time unit (seconds or milliseconds)."""
        unit = self.unit_var.get()
        if unit == "seconds":
            max_value = self.audio_duration
            resolution = 0.001  # Allows sub-second precision
        else:  # milliseconds
            max_value = self.audio_duration * 1000
            resolution = 1  # 1ms resolution

        self.start_slider.config(from_=0, to=max_value, resolution=resolution)
        self.end_slider.config(from_=0, to=max_value, resolution=resolution)

        # Reset slider values to max range
        self.start_slider.set(0)
        self.end_slider.set(max_value)

        self.update_waveform(None)

    def update_waveform(self, event):
        """Update the waveform visualization based on the selected trim points."""
        if self.audio is None:
            return

        unit = self.unit_var.get()
        start_time = self.start_var.get() if unit == "seconds" else self.start_var.get() / 1000
        end_time = self.end_var.get() if unit == "seconds" else self.end_var.get() / 1000

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        trimmed_audio = self.audio[start_ms:end_ms]

        # Convert audio to numpy array and normalize it
        samples = np.array(trimmed_audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.max(np.abs(samples)) if len(samples) > 0 else samples  # Normalize

        # Generate time axis for waveform
        time_axis = np.linspace(start_time, end_time, num=len(samples))

        # Display trimmed waveform
        self.ax_waveform.clear()
        self.ax_waveform.plot(time_axis, samples, color='blue')
        self.ax_waveform.set_title("Trimmed Audio Waveform")
        self.ax_waveform.set_xlabel(f"Time ({unit})")
        self.ax_waveform.set_ylabel("Amplitude")

        # Update frequency spectrum
        if len(samples) > 0:
            fft_data = np.abs(fft(samples))[:len(samples) // 2]
            freqs = np.fft.fftfreq(len(samples), d=1 / self.sample_rate)[:len(samples) // 2]

            self.ax_spectrum.clear()
            self.ax_spectrum.plot(freqs, fft_data, color='red')
            self.ax_spectrum.set_title("Frequency Spectrum")
            self.ax_spectrum.set_xlabel("Frequency (Hz)")
            self.ax_spectrum.set_ylabel("Magnitude")

        self.canvas.draw()

    def trim_audio(self):
        """Trim the audio based on slider values and save the output."""
        if self.audio is None:
            messagebox.showerror("Error", "No audio file selected.")
            return

        unit = self.unit_var.get()
        start_ms = int(self.start_var.get() * 1000 if unit == "seconds" else self.start_var.get())
        end_ms = int(self.end_var.get() * 1000 if unit == "seconds" else self.end_var.get())

        if start_ms >= end_ms:
            messagebox.showerror("Error", "End time must be greater than start time.")
            return

        trimmed_audio = self.audio[start_ms:end_ms]

        # Ask for save location
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 Files", "*.mp3"), ("WAV Files", "*.wav"), ("OGG Files", "*.ogg")]
        )
        if not output_path:
            return

        trimmed_audio.export(output_path, format=output_path.split('.')[-1])
        messagebox.showinfo("Success", f"Trimmed audio saved as:\n{output_path}")

# Run the GUI
root = tk.Tk()
app = AudioTrimmer(root)
root.mainloop()
