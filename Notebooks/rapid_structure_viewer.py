import tkinter as tk
from tkinter import filedialog
from pymatgen.core import Structure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from flask import Flask, render_template_string
import threading
import webbrowser

app = Flask(__name__)

# Global variable to store the current HTML content
current_html = ""

@app.route("/")
def index():
    global current_html
    return render_template_string(current_html)

def start_flask():
    app.run(debug=False, port=5000, use_reloader=False)

class StructureVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Pymatgen Structure Visualizer")

        self.load_button = tk.Button(root, text="Load Structures", command=self.load_structures)
        self.load_button.pack()

        self.next_button = tk.Button(root, text="Next", command=self.next_structure)
        self.next_button.pack()

        self.prev_button = tk.Button(root, text="Previous", command=self.prev_structure)
        self.prev_button.pack()

        self.structure_label = tk.Label(root, text="No structure loaded")
        self.structure_label.pack()

        self.structures = []
        self.current_index = -1

        # Start Flask in a separate thread
        threading.Thread(target=start_flask, daemon=True).start()

    def load_structures(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Structure files", "*.cif *.vasp *.poscar *.json *.mson *.yaml")])
        self.structures = [Structure.from_file(file_path) for file_path in file_paths]
        self.current_index = 0 if self.structures else -1
        self.display_structure()

    def next_structure(self):
        if self.structures and self.current_index < len(self.structures) - 1:
            self.current_index += 1
            self.display_structure()

    def prev_structure(self):
        if self.structures and self.current_index > 0:
            self.current_index -= 1
            self.display_structure()

    def display_structure(self):
        if self.structures and self.current_index != -1:
            structure = self.structures[self.current_index]
            self.structure_label.config(text=f"Structure {self.current_index + 1} of {len(self.structures)}")

            fig = self.plot_structure(structure)
            global current_html
            current_html = pio.to_html(fig, full_html=True)
            webbrowser.open("http://localhost:5000")

    def plot_structure(self, structure):
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

        for site in structure:
            fig.add_trace(
                go.Scatter3d(
                    x=[site.coords[0]],
                    y=[site.coords[1]],
                    z=[site.coords[2]],
                    mode='markers',
                    marker=dict(size=10),
                    name=site.specie.symbol
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        return fig

if __name__ == "__main__":
    root = tk.Tk()
    app = StructureVisualizer(root)
    root.mainloop()
