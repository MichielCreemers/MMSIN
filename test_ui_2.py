import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

class ExampleWindow:
    def __init__(self, master):
        self.master = master
        master.title("Test")

        # Apply a themed style
        self.style = ttk.Style(master)
        self.style.theme_use("clam")  # Choose a modern-looking theme

        # Set dark theme colors
        self.style.configure('.', background='black', foreground='white')
        self.style.map('.', background=[('active', '#007acc')])

        # Create a frame for layout
        self.gui_layout = ttk.Frame(master, padding="20", style='TFrame')
        self.gui_layout.pack()

        # Create a file-chooser widget for point cloud file
        self.pcfileedit = ttk.Entry(self.gui_layout, style='TEntry')
        self.pcfileedit.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.pcfilebutton = ttk.Button(self.gui_layout, text="...", command=self._on_pc_button, style='TButton')
        self.pcfilebutton.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(self.gui_layout, text="Select point cloud file", style='TLabel').grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create a file-chooser widget for model file
        self.modelfileedit = ttk.Entry(self.gui_layout, style='TEntry')
        self.modelfileedit.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.modelfilebutton = ttk.Button(self.gui_layout, text="...", command=self._on_model_button, style='TButton')
        self.modelfilebutton.grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(self.gui_layout, text="Select model file", style='TLabel').grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # Number editors for the x and y projections
        self.x_proj = ttk.Entry(self.gui_layout, style='TEntry')
        self.x_proj.grid(row=2, column=1, padx=5, pady=5)

        self.y_proj = ttk.Entry(self.gui_layout, style='TEntry')
        self.y_proj.grid(row=2, column=3, padx=5, pady=5)

        ttk.Label(self.gui_layout, text="Number of x-projections", style='TLabel').grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(self.gui_layout, text="Number of y-projections", style='TLabel').grid(row=2, column=2, padx=5, pady=5, sticky="w")

        # Create a progress bar
        self.progress_label = ttk.Label(self.gui_layout, text="Add point cloud", style='TLabel')
        self.progress_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        self.progress = ttk.Progressbar(self.gui_layout, orient="horizontal", length=200, mode="determinate", style='TProgressbar')
        self.progress.grid(row=3, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Create confirmation button
        self.ok_button = ttk.Button(self.gui_layout, text="Calculate quality", command=self._on_ok, style='TButton')
        self.ok_button.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Configure weights for layout expansion
        for i in range(5):
            self.gui_layout.rowconfigure(i, weight=1)
        for j in range(4):
            self.gui_layout.columnconfigure(j, weight=1)

    def _on_pc_button(self):
        filename = filedialog.askopenfilename()
        self.pcfileedit.delete(0, tk.END)
        self.pcfileedit.insert(0, filename)

    def _on_model_button(self):
        filename = filedialog.askopenfilename()
        self.modelfileedit.delete(0, tk.END)
        self.modelfileedit.insert(0, filename)

    def _on_ok(self):
        # Implement your logic here for the ok button
        pass

# Create the main window using Tkinter
root = tk.Tk()
app = ExampleWindow(root)
root.mainloop()
