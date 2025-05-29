import json
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from collections import defaultdict
import os
from datetime import datetime

class VisPipelineEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("VIS Pipeline Editor")
        self.root.geometry("1200x800")
        
        self.pipeline = []
        self.current_module_index = None
        
        self.create_ui()
    
    def create_ui(self):
        # Create main frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for pipeline modules
        self.left_frame = ttk.LabelFrame(self.main_frame, text="Pipeline Modules")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel for editor
        self.right_frame = ttk.LabelFrame(self.main_frame, text="Module Editor")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom panel for generated code
        self.bottom_frame = ttk.LabelFrame(self.root, text="Generated Python Code")
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create pipeline listbox
        self.create_pipeline_listbox()
        
        # Create module editor
        self.create_module_editor()
        
        # Create code preview
        self.create_code_preview()
        
        # Create menu
        self.create_menu()
    
    def create_pipeline_listbox(self):
        # Frame for list and buttons
        list_frame = ttk.Frame(self.left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Pipeline module list
        self.pipeline_list = tk.Listbox(list_frame, width=40, height=20)
        self.pipeline_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.pipeline_list.bind('<<ListboxSelect>>', self.on_module_select)
        
        # Scrollbar for list
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pipeline_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.pipeline_list.config(yscrollcommand=scrollbar.set)
        
        # Buttons for pipeline management
        btn_frame = ttk.Frame(self.left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add Module", command=self.add_module).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Module", command=self.delete_module).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Move Up", command=self.move_module_up).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Move Down", command=self.move_module_down).pack(side=tk.LEFT, padx=5)
    
    def create_module_editor(self):
        # Module properties
        properties_frame = ttk.Frame(self.right_frame)
        properties_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Module name
        ttk.Label(properties_frame, text="Module Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.module_name = ttk.Entry(properties_frame, width=30)
        self.module_name.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Mode
        ttk.Label(properties_frame, text="Mode:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="live")
        mode_combo = ttk.Combobox(properties_frame, textvariable=self.mode_var, values=["live", "single", "batch"])
        mode_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Level
        ttk.Label(properties_frame, text="Level:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.level_var = tk.IntVar(value=0)
        level_spinner = ttk.Spinbox(properties_frame, from_=0, to=10, textvariable=self.level_var, width=5)
        level_spinner.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Input From
        ttk.Label(properties_frame, text="Input From:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.input_from = ttk.Entry(properties_frame, width=30)
        self.input_from.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Input Type
        ttk.Label(properties_frame, text="Input Type:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.input_type = ttk.Entry(properties_frame, width=30)
        self.input_type.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Prompt
        ttk.Label(properties_frame, text="Prompt:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.prompt = ttk.Entry(properties_frame, width=30)
        self.prompt.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # Save button
        ttk.Button(properties_frame, text="Save Module", command=self.save_module).grid(row=6, column=0, columnspan=2, pady=10)
        
        # Disable editor initially
        self.set_editor_state(tk.DISABLED)
    
    def create_code_preview(self):
        # Code preview text area
        self.code_preview = scrolledtext.ScrolledText(self.bottom_frame, width=80, height=15, wrap=tk.WORD)
        self.code_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Button frame for code actions
        code_btn_frame = ttk.Frame(self.bottom_frame)
        code_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(code_btn_frame, text="Generate Code", command=self.generate_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(code_btn_frame, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(code_btn_frame, text="Save Python File", command=self.save_python_file).pack(side=tk.LEFT, padx=5)
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Pipeline", command=self.new_pipeline)
        file_menu.add_command(label="Open JSON", command=self.open_json)
        file_menu.add_command(label="Save JSON", command=self.save_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def update_pipeline_list(self):
        self.pipeline_list.delete(0, tk.END)
        for i, module in enumerate(self.pipeline):
            self.pipeline_list.insert(tk.END, f"Level {module['level']}: {module['module']}")
        self.generate_code()
    
    def set_editor_state(self, state):
        for widget in [self.module_name, self.input_from, self.input_type, self.prompt]:
            widget.config(state=state)
    
    def load_module_to_editor(self, module):
        self.set_editor_state(tk.NORMAL)
        
        self.module_name.delete(0, tk.END)
        self.module_name.insert(0, module.get('module', ''))
        
        self.mode_var.set(module.get('mode', 'live'))
        
        self.level_var.set(module.get('level', 0))
        
        self.input_from.delete(0, tk.END)
        self.input_from.insert(0, module.get('input_from', ''))
        
        self.input_type.delete(0, tk.END)
        self.input_type.insert(0, module.get('input_type', ''))
        
        self.prompt.delete(0, tk.END)
        self.prompt.insert(0, module.get('prompt', ''))
    
    def on_module_select(self, event):
        selection = self.pipeline_list.curselection()
        if selection:
            self.current_module_index = selection[0]
            self.load_module_to_editor(self.pipeline[self.current_module_index])
    
    def add_module(self):
        new_module = {
            "module": f"new_module_{len(self.pipeline)}",
            "mode": "live",
            "level": len(self.pipeline)
        }
        self.pipeline.append(new_module)
        self.update_pipeline_list()
        # Select the new module
        self.pipeline_list.selection_clear(0, tk.END)
        self.pipeline_list.selection_set(len(self.pipeline) - 1)
        self.on_module_select(None)
    
    def delete_module(self):
        if self.current_module_index is not None:
            del self.pipeline[self.current_module_index]
            self.update_pipeline_list()
            self.current_module_index = None
            self.set_editor_state(tk.DISABLED)
    
    def save_module(self):
        if self.current_module_index is not None:
            self.pipeline[self.current_module_index] = {
                "module": self.module_name.get(),
                "mode": self.mode_var.get(),
                "level": self.level_var.get(),
                "input_from": self.input_from.get() if self.input_from.get() else None,
                "input_type": self.input_type.get() if self.input_type.get() else None
            }
            
            # Add prompt only if it's provided
            if self.prompt.get():
                self.pipeline[self.current_module_index]["prompt"] = self.prompt.get()
            
            # Clean up None values
            self.pipeline[self.current_module_index] = {k: v for k, v in self.pipeline[self.current_module_index].items() if v is not None}
            
            self.update_pipeline_list()
    
    def move_module_up(self):
        if self.current_module_index is not None and self.current_module_index > 0:
            self.pipeline[self.current_module_index], self.pipeline[self.current_module_index - 1] = \
                self.pipeline[self.current_module_index - 1], self.pipeline[self.current_module_index]
            self.update_pipeline_list()
            self.pipeline_list.selection_clear(0, tk.END)
            self.pipeline_list.selection_set(self.current_module_index - 1)
            self.current_module_index -= 1
    
    def move_module_down(self):
        if self.current_module_index is not None and self.current_module_index < len(self.pipeline) - 1:
            self.pipeline[self.current_module_index], self.pipeline[self.current_module_index + 1] = \
                self.pipeline[self.current_module_index + 1], self.pipeline[self.current_module_index]
            self.update_pipeline_list()
            self.pipeline_list.selection_clear(0, tk.END)
            self.pipeline_list.selection_set(self.current_module_index + 1)
            self.current_module_index += 1
    
    def generate_pipeline_code(self):
        """
        Generate Python code for the VIS pipeline from the current configuration.
        """
        # Group modules by level
        modules_by_level = defaultdict(list)
        all_modules = []
        
        for module_config in self.pipeline:
            level = module_config.get("level", 0)
            modules_by_level[level].append(module_config)
            all_modules.append(module_config["module"])
        
        # Generate code
        code = [
            "# Generated by VIS Pipeline Editor",
            f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# User: {os.getlogin()}",
            "",
            "from sil_sdk.modules.vis import VISModule",
            "",
            "# Initialize the VIS module with WebSocket connection",
            "vis = VISModule(\"ws://<server-ip>:8765\")",
            "",
            "# Load all modules",
            f"modules = {all_modules}",
            "vis.load(modules)",
            ""
        ]
        
        # Dictionary to track results from each module
        results_vars = {}
        
        # Process each level
        for level in sorted(modules_by_level.keys()):
            code.append(f"# Level {level}")
            
            for module_config in modules_by_level[level]:
                module_name = module_config["module"]
                mode = module_config.get("mode", "live")
                input_from = module_config.get("input_from", None)
                input_type = module_config.get("input_type", None)
                prompt = module_config.get("prompt", None)
                
                # Generate run statement
                run_params = []
                
                # Handle different types of inputs
                if input_from and input_type:
                    if input_from == "camera" and input_type == "camera":
                        run_params.append("input_source=\"camera\"")
                    elif input_from in results_vars:
                        # Get data from previous module
                        input_var = f"{input_from.lower()}_result.get(\"{input_type}\")"
                        var_name = input_type.lower()
                        code.append(f"{var_name} = {input_var}")
                        run_params.append(f"{var_name}={var_name}")
                
                # Add prompt if provided
                if prompt:
                    run_params.append(f"prompt=\"{prompt}\"")
                    
                # Generate the run statement
                params_str = ", ".join(run_params)
                code.append(f"vis.run(\"{module_name}\", {params_str})")
                result_var = f"{module_name.lower()}_result"
                code.append(f"{result_var} = vis.get_result(\"{module_name}\")")
                results_vars[module_name] = result_var
                code.append("")
            
        # Add comments for example usage
        code.append("# Now you can use the results from all modules")
        code.append("# For example:")
        
        for module_config in self.pipeline:
            module_name = module_config["module"].lower()
            if "gdino" in module_name.lower():
                code.append(f"# bboxes = {module_name}_result.get(\"BBOX\")")
            elif "grasp" in module_name.lower():
                code.append(f"# grasp_poses = {module_name}_result.get(\"grasp_poses\")")
            elif "llava" in module_name.lower():
                code.append(f"# description = {module_name}_result.get(\"text\")")
            elif "pose" in module_name.lower():
                code.append(f"# pose = {module_name}_result.get(\"pose\")")
            else:
                code.append(f"# result = {module_name}_result.get(\"output\")")
        
        return "\n".join(code)
    
    def generate_code(self):
        if not self.pipeline:
            self.code_preview.delete(1.0, tk.END)
            self.code_preview.insert(tk.END, "# Add modules to generate code")
            return
        
        code = self.generate_pipeline_code()
        self.code_preview.delete(1.0, tk.END)
        self.code_preview.insert(tk.END, code)
    
    def copy_to_clipboard(self):
        code = self.code_preview.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        messagebox.showinfo("Copied", "Code copied to clipboard!")
    
    def save_python_file(self):
        if not self.pipeline:
            messagebox.showwarning("Warning", "No code to save. Add modules first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            title="Save Python Code"
        )
        
        if file_path:
            with open(file_path, 'w') as file:
                file.write(self.code_preview.get(1.0, tk.END))
            messagebox.showinfo("Saved", f"Code saved to {file_path}")
    
    def new_pipeline(self):
        if messagebox.askyesno("New Pipeline", "Are you sure you want to create a new pipeline? Any unsaved changes will be lost."):
            self.pipeline = []
            self.current_module_index = None
            self.update_pipeline_list()
            self.set_editor_state(tk.DISABLED)
    
    def open_json(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Open Pipeline JSON"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.pipeline = json.load(file)
                self.update_pipeline_list()
                messagebox.showinfo("Loaded", f"Pipeline loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load pipeline: {str(e)}")
    
    def save_json(self):
        if not self.pipeline:
            messagebox.showwarning("Warning", "No pipeline to save. Add modules first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Pipeline JSON"
        )
        
        if file_path:
            with open(file_path, 'w') as file:
                json.dump(self.pipeline, file, indent=2)
            messagebox.showinfo("Saved", f"Pipeline saved to {file_path}")
    
    def show_about(self):
        messagebox.showinfo(
            "About VIS Pipeline Editor",
            "VIS Pipeline Editor\n\n"
            "A tool for creating and editing VIS module pipelines for the SIL SDK.\n\n"
            "Created for Hritikshah02"
        )
    
    def show_docs(self):
        messagebox.showinfo(
            "Documentation",
            "VIS Pipeline Editor Documentation\n\n"
            "1. Create a new pipeline or open an existing JSON pipeline\n"
            "2. Add modules and configure their properties\n"
            "3. Set the level to determine execution order\n"
            "4. Generate Python code and save it to a file\n\n"
            "For more information, visit: https://pypi.org/project/sil-sdk/"
        )


def main():
    root = tk.Tk()
    app = VisPipelineEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()