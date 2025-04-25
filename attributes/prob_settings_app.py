import os
import json
import tkinter as ttk
import tkinter.messagebox as messagebox

# Hardcoded directory path
TXT_DIRECTORY = "attributes/attributes"

# File to save file probabilities
PROB_FILE = "attributes/attributes/file_probabilities.json"

ATTRIBUTE_CLASH_FILE = "attributes/attributes/attribute_clashes.json"

# Ensure TXT_DIRECTORY exists
if not os.path.exists(TXT_DIRECTORY):
    os.makedirs(TXT_DIRECTORY)

# Load file probabilities
if os.path.exists(PROB_FILE):
    with open(PROB_FILE, "r") as f:
        file_probabilities = json.load(f)
else:
    file_probabilities = {}


# Enforce numeric input with at most one decimal place
def validate_numeric_input(P):
    if P == "" or (P.replace(".", "", 1).isdigit() and P.count(".") <= 1):
        try:
            if P == "" or (0 <= float(P) <= 100):
                if len(P.split(".")[-1]) <= 8:
                    return True
        except ValueError:
            return False
    return False


# Save file probabilities
def save_file_probabilities():
    with open(PROB_FILE, "w") as f:
        json.dump(file_probabilities, f, indent=4)


# Edit probabilities for files
def edit_file_probabilities():
    def save_changes():
        for i, file in enumerate(files):
            file_probabilities[file] = round(file_prob_vars[i].get() / 100, 10)
        save_file_probabilities()
        on_close()

    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_close():
        editor_window.unbind_all("<MouseWheel>")
        editor_window.destroy()

    def focus_next(event):
        widget = event.widget
        idx = prob_entries.index(widget) if widget in prob_entries else -1
        if idx >= 0 and idx < len(prob_entries) - 1:
            next_entry = prob_entries[idx + 1]
            next_entry.focus()
            next_entry.icursor("end")
        return "break"

    def focus_prev(event):
        widget = event.widget
        idx = prob_entries.index(widget) if widget in prob_entries else -1
        if idx > 0:
            prev_entry = prob_entries[idx - 1]
            prev_entry.focus()
            prev_entry.icursor("end")
        return "break"

    editor_window = ttk.Toplevel(root)
    editor_window.title("Edit Probability for Each Text File")
    editor_window.geometry("600x500")

    main_frame = ttk.Frame(editor_window)
    main_frame.pack(fill="both", expand=True)

    canvas = ttk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=ttk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Bind mouse wheel scrolling
    editor_window.bind_all("<MouseWheel>", on_mousewheel)

    header_frame = ttk.Frame(scrollable_frame)
    header_frame.pack(fill="x", pady=5)

    file_label = ttk.Label(header_frame, text="File Name", width=52, anchor="w", font=("Arial", 10, "bold"))
    file_label.pack(side="left", padx=5)

    prob_label = ttk.Label(header_frame, text="Probability (%)", width=15, anchor="w", font=("Arial", 10, "bold"))
    prob_label.pack(side="left", padx=5)

    file_prob_vars = []
    prob_entries = []

    for file in files:
        file_var = ttk.StringVar(value=file)
        prob_var = ttk.DoubleVar(value=round(file_probabilities.get(file, 0) * 100, 8))
        file_prob_vars.append(prob_var)

        row_frame = ttk.Frame(scrollable_frame)
        row_frame.pack(fill="x", pady=2)

        file_label = ttk.Label(row_frame, text=file, width=60, anchor="w")
        file_label.pack(side="left", padx=5)

        prob_entry = ttk.Entry(row_frame, textvariable=prob_var, width=15, validate="key")
        prob_entry.pack(side="left")
        prob_entry["validatecommand"] = (editor_window.register(validate_numeric_input), "%P")
        prob_entry.bind("<Up>", focus_prev)
        prob_entry.bind("<Down>", focus_next)

        prob_entries.append(prob_entry)

    # Add save button at the bottom of the main window
    save_button = ttk.Button(editor_window, text="Save Changes", command=save_changes)
    save_button.pack(pady=10, side="bottom")

    # Bind close event
    editor_window.protocol("WM_DELETE_WINDOW", on_close)


def edit_file(file_path):
    def save_changes():
        # Convert probabilities to proportions and round to 4 decimal places
        rounded_probs = [round(prob.get() / 100, 10) for prob in probabilities]

        # Calculate the total after rounding
        rounded_total = sum(rounded_probs)

        # Calculate the discrepancy
        discrepancy = round(1.0 - rounded_total, 10)

        if abs(discrepancy) > 0.0:  # Only adjust if there's a discrepancy
            # Determine whether to add or subtract based on the discrepancy
            if discrepancy > 0:
                # Add to the highest probability
                max_index = max(range(len(rounded_probs)), key=lambda i: rounded_probs[i])
                rounded_probs[max_index] += discrepancy
            elif discrepancy < 0:
                # Subtract from the highest probability
                max_index = max(range(len(rounded_probs)), key=lambda i: rounded_probs[i])
                rounded_probs[max_index] += discrepancy  # Subtract by adding a negative value

        # Write adjusted probabilities to the file
        with open(file_path, "w") as f:
            for option, prob in zip(options, rounded_probs):
                f.write(f"{option.get()},{prob:.10f}\n")

        on_close()

    def update_total_label():
        total = sum(prob.get() for prob in probabilities)
        if round(abs(total - 100.0), 8) <= 0.01:
            total_label.config(text="Probabilities add up to 100%", fg="green")
            save_button.config(state="normal")
        else:
            total_label.config(text=f"Total is {total:.2f}%. Please adjust.", fg="red")
            save_button.config(state="disabled")

    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_close():
        editor_window.unbind_all("<MouseWheel>")
        editor_window.destroy()

    def initialize_probabilities():
        total_options = len(lines)

        if total_options == 0:
            return []

        initial_prob = round(100.0 / total_options, 8)
        probabilities = [initial_prob] * total_options

        total = sum(probabilities)
        discrepancy = round(100.0 - total, 8)

        if abs(discrepancy) > 0.0:
            max_index = max(range(total_options), key=lambda i: probabilities[i])
            probabilities[max_index] += discrepancy

        return probabilities

    def focus_next(event):
        widget = event.widget
        idx = prob_entries.index(widget) if widget in prob_entries else -1
        if idx >= 0 and idx < len(prob_entries) - 1:
            next_entry = prob_entries[idx + 1]
            next_entry.focus()
            next_entry.icursor("end")
        return "break"

    def focus_prev(event):
        widget = event.widget
        idx = prob_entries.index(widget) if widget in prob_entries else -1
        if idx > 0:
            prev_entry = prob_entries[idx - 1]
            prev_entry.focus()
            prev_entry.icursor("end")
        return "break"

    editor_window = ttk.Toplevel(root)
    editor_window.title(f"Editing File's Attribute Probabilites: {file_path}")
    editor_window.geometry("600x500")

    frame = ttk.Frame(editor_window)
    frame.pack(fill="both", expand=True)

    header_frame = ttk.Frame(frame)
    header_frame.pack(fill="x", pady=5)

    attr_label = ttk.Label(header_frame, text="Attribute", width=52, anchor="w", font=("Arial", 10, "bold"))
    attr_label.pack(side="left", padx=5)

    prob_label = ttk.Label(header_frame, text="Probability (%)", width=15, anchor="w", font=("Arial", 10, "bold"))
    prob_label.pack(side="left", padx=5)

    canvas = ttk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient=ttk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    editor_window.bind_all("<MouseWheel>", on_mousewheel)

    options = []
    probabilities = []
    prob_entries = []

    with open(file_path, "r") as f:
        lines = f.readlines()
        rounded_probs = initialize_probabilities()

        for i, line in enumerate(lines):
            line = line.strip()
            if "," in line:
                option, prob = line.split(",")
                prob = float(prob) * 100
            else:
                option, prob = line, rounded_probs[i]

            option_var = ttk.StringVar(value=option)
            prob_var = ttk.DoubleVar(value=round(prob, 8))
            options.append(option_var)
            probabilities.append(prob_var)

            row_frame = ttk.Frame(scrollable_frame)
            row_frame.pack(fill="x", pady=2)

            option_label = ttk.Label(row_frame, text=option, width=60, anchor="w")
            option_label.pack(side="left", padx=5)

            prob_entry = ttk.Entry(row_frame, textvariable=prob_var, width=15, validate="key")
            prob_entry.pack(side="left")
            prob_entry.bind("<KeyRelease>", lambda event: update_total_label())
            prob_entry.bind("<Up>", focus_prev)
            prob_entry.bind("<Down>", focus_next)
            prob_entry["validatecommand"] = (editor_window.register(validate_numeric_input), "%P")

            prob_entries.append(prob_entry)

    save_button = ttk.Button(editor_window, text="Save Changes", command=save_changes, state="disabled")
    save_button.pack(pady=10)

    total_label = ttk.Label(editor_window, text="Total: 0%")
    total_label.pack(pady=5)

    editor_window.protocol("WM_DELETE_WINDOW", on_close)

    update_total_label()


def create_file():
    def save_new_file():
        new_file_name = file_name_var.get().strip()
        if not new_file_name.endswith(".txt"):
            new_file_name += ".txt"

        new_file_path = os.path.join(TXT_DIRECTORY, new_file_name)
        if os.path.exists(new_file_path):
            messagebox.showerror("Error", "File already exists!")
        else:
            with open(new_file_path, "w") as f:
                f.write("")  # Create an empty file
            files.append(new_file_name)
            file_listbox.insert(ttk.END, new_file_name)
            messagebox.showinfo("Success", f"File '{new_file_name}' created successfully.")
            create_window.destroy()

    create_window = ttk.Toplevel(root)
    create_window.title("Create New File")
    create_window.geometry("300x150")

    ttk.Label(create_window, text="Enter file name:").pack(pady=10)

    file_name_var = ttk.StringVar()
    ttk.Entry(create_window, textvariable=file_name_var, width=30).pack(pady=5)

    ttk.Button(create_window, text="Create File", command=save_new_file).pack(pady=10)


def edit_file_content(file_path):
    def save_file_content():
        with open(file_path, "w") as f:
            f.write(text_editor.get("1.0", ttk.END).strip())
        messagebox.showinfo("Success", f"File '{os.path.basename(file_path)}' saved successfully.")
        editor_window.destroy()

    editor_window = ttk.Toplevel(root)
    editor_window.title(f"Editing: {os.path.basename(file_path)}")
    editor_window.geometry("600x500")

    frame = ttk.Frame(editor_window)
    frame.pack(fill="both", expand=True)

    scrollbar = ttk.Scrollbar(frame, orient=ttk.VERTICAL)
    scrollbar.pack(side="right", fill="y")

    text_editor = ttk.Text(frame, wrap="word", yscrollcommand=scrollbar.set)
    text_editor.pack(fill="both", expand=True, side="left")

    scrollbar.config(command=text_editor.yview)

    with open(file_path, "r") as f:
        text_editor.insert("1.0", f.read())

    save_button = ttk.Button(editor_window, text="Save Changes", command=save_file_content)
    save_button.pack(pady=5)


def delete_file():
    selected_index = file_listbox.curselection()
    if not selected_index:
        messagebox.showerror("Error", "No file selected!")
        return

    selected_file = file_listbox.get(selected_index)
    file_path = os.path.join(TXT_DIRECTORY, selected_file)

    if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete '{selected_file}'?"):
        os.remove(file_path)
        file_listbox.delete(selected_index)
        files.remove(selected_file)


def show_context_menu(event):
    widget = event.widget

    if widget == file_listbox:
        selected_index = file_listbox.nearest(event.y)
        file_listbox.selection_clear(0, ttk.END)
        file_listbox.selection_set(selected_index)
        file_context_menu.tk_popup(event.x_root, event.y_root)


# Function to declare attribute clashes
def declare_attribute_clashes():
    def choose_primary_file():
        selected_index = primary_file_listbox.curselection()
        if selected_index:
            selected_file = primary_file_listbox.get(selected_index)
            primary_file_var.set(selected_file)
            load_attributes(selected_file, primary_attributes_frame, primary_check_vars, primary_canvas)

    def choose_secondary_file():
        selected_index = secondary_file_listbox.curselection()
        if selected_index:
            selected_file = secondary_file_listbox.get(selected_index)
            secondary_file_var.set(selected_file)
            load_attributes(selected_file, secondary_attributes_frame, secondary_check_vars, secondary_canvas)

    def load_attributes(file, frame, check_vars, canvas):
        for widget in frame.winfo_children():
            widget.destroy()

        check_vars.clear()

        if file:
            file_path = os.path.join(TXT_DIRECTORY, file)
            with open(file_path, "r") as f:
                attributes = [line.split(",")[0].strip() for line in f if line.strip()]

            select_all_var = ttk.BooleanVar()

            def toggle_select_all():
                for var in check_vars.values():
                    var.set(select_all_var.get())

            select_all_check = ttk.Checkbutton(
                frame,
                text="Select All",
                variable=select_all_var,
                command=toggle_select_all,
            )
            select_all_check.pack(anchor="w", pady=2)

            for attr in attributes:
                var = ttk.BooleanVar()
                check_vars[attr] = var
                check = ttk.Checkbutton(frame, text=attr, variable=var)
                check.pack(anchor="w", pady=2)

        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def bind_mousewheel(canvas):
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _: canvas.bind("<MouseWheel>", on_mousewheel))
        canvas.bind("<Leave>", lambda _: canvas.unbind("<MouseWheel>"))

    def save_attribute_clashes():
        primary_file = primary_file_var.get()
        secondary_file = secondary_file_var.get()

        if primary_file == secondary_file:
            messagebox.showerror("Error", "Primary and Secondary files cannot be the same.")
            return

        if not primary_file or not secondary_file:
            messagebox.showerror("Error", "Both primary and secondary files must be selected before saving.")
            return

        primary_checked = [attr for attr, var in primary_check_vars.items() if var.get()]
        secondary_checked = [attr for attr, var in secondary_check_vars.items() if var.get()]

        if not primary_checked:
            messagebox.showerror("Error", f"No attributes selected for the Fixed Attribute(s) file: {primary_file}")
            return

        if not secondary_checked:
            messagebox.showerror(
                "Error", f"No attributes selected for the Secondary Attribute(s) file: {secondary_file}"
            )
            return

        # Use [all] if all attributes are selected
        primary_checked = (
            [var for var in primary_check_vars] if len(primary_checked) == len(primary_check_vars) else primary_checked
        )
        secondary_checked = (
            [var for var in secondary_check_vars]
            if len(secondary_checked) == len(secondary_check_vars)
            else secondary_checked
        )

        clash_entry = {
            "fixed_file": primary_file,
            "fixed_attributes": primary_checked,
            "secondary_file": secondary_file,
            "secondary_attributes": secondary_checked,
        }

        if not os.path.exists(ATTRIBUTE_CLASH_FILE):
            with open(ATTRIBUTE_CLASH_FILE, "w") as f:
                json.dump([], f)

        with open(ATTRIBUTE_CLASH_FILE, "r+") as f:
            try:
                clashes = json.load(f)
            except json.JSONDecodeError:
                clashes = []

            clashes.append(clash_entry)
            f.seek(0)
            json.dump(clashes, f, indent=4)

    clash_window = ttk.Toplevel(root)  # Use Toplevel to create a new child window
    clash_window.title("Declare Attribute Clashes")
    clash_window.geometry("800x600")

    content_frame = ttk.Frame(clash_window)
    content_frame.pack(fill="both", expand=True, padx=10, pady=5)

    primary_file_frame = ttk.Frame(content_frame)
    primary_file_frame.pack(side="left", fill="both", expand=True, padx=10, pady=5)
    primary_file_frame.pack_propagate(False)

    primary_file_label = ttk.Label(primary_file_frame, text="Fixed Attribute(s)", font=("Arial", 12, "bold"))
    primary_file_label.pack(pady=5)

    primary_file_var = ttk.StringVar()
    primary_file_entry = ttk.Entry(primary_file_frame, textvariable=primary_file_var, state="readonly", width=40)
    primary_file_entry.pack(pady=5)

    primary_file_listbox_frame = ttk.Frame(primary_file_frame)
    primary_file_listbox_frame.pack(fill="both", expand=False, pady=5)

    primary_file_listbox = ttk.Listbox(primary_file_listbox_frame, height=8)
    primary_file_listbox.pack(side="left", fill="both", expand=True)

    primary_file_scrollbar = ttk.Scrollbar(
        primary_file_listbox_frame, orient="vertical", command=primary_file_listbox.yview
    )
    primary_file_scrollbar.pack(side="right", fill="y")

    primary_file_listbox.configure(yscrollcommand=primary_file_scrollbar.set)

    for file in files:
        primary_file_listbox.insert(ttk.END, file)

    primary_file_button = ttk.Button(primary_file_frame, text="Choose File", command=choose_primary_file)
    primary_file_button.pack(pady=5)

    primary_attributes_scrollable_frame = ttk.Frame(primary_file_frame)
    primary_attributes_scrollable_frame.pack(fill="both", expand=True)

    primary_canvas = ttk.Canvas(primary_attributes_scrollable_frame)
    primary_scrollbar = ttk.Scrollbar(
        primary_attributes_scrollable_frame, orient="vertical", command=primary_canvas.yview
    )
    primary_canvas.configure(yscrollcommand=primary_scrollbar.set)

    primary_attributes_frame = ttk.Frame(primary_canvas)

    primary_canvas.create_window((0, 0), window=primary_attributes_frame, anchor="nw")
    primary_scrollbar.pack(side="right", fill="y")
    primary_canvas.pack(side="left", fill="both", expand=True)

    bind_mousewheel(primary_canvas)

    primary_check_vars = {}

    secondary_file_frame = ttk.Frame(content_frame)
    secondary_file_frame.pack(side="right", fill="both", expand=True, padx=10, pady=5)
    secondary_file_frame.pack_propagate(False)

    secondary_file_label = ttk.Label(secondary_file_frame, text="Secondary Attribute(s)", font=("Arial", 12, "bold"))
    secondary_file_label.pack(pady=5)

    secondary_file_var = ttk.StringVar()
    secondary_file_entry = ttk.Entry(secondary_file_frame, textvariable=secondary_file_var, state="readonly", width=40)
    secondary_file_entry.pack(pady=5)

    secondary_file_listbox_frame = ttk.Frame(secondary_file_frame)
    secondary_file_listbox_frame.pack(fill="both", expand=False, pady=5)

    secondary_file_listbox = ttk.Listbox(secondary_file_listbox_frame, height=8)
    secondary_file_listbox.pack(side="left", fill="both", expand=True)

    secondary_file_scrollbar = ttk.Scrollbar(
        secondary_file_listbox_frame, orient="vertical", command=secondary_file_listbox.yview
    )
    secondary_file_scrollbar.pack(side="right", fill="y")

    secondary_file_listbox.configure(yscrollcommand=secondary_file_scrollbar.set)

    for file in files:
        secondary_file_listbox.insert(ttk.END, file)

    secondary_file_button = ttk.Button(secondary_file_frame, text="Choose File", command=choose_secondary_file)
    secondary_file_button.pack(pady=5)

    secondary_attributes_scrollable_frame = ttk.Frame(secondary_file_frame)
    secondary_attributes_scrollable_frame.pack(fill="both", expand=True)

    secondary_canvas = ttk.Canvas(secondary_attributes_scrollable_frame)
    secondary_scrollbar = ttk.Scrollbar(
        secondary_attributes_scrollable_frame, orient="vertical", command=secondary_canvas.yview
    )
    secondary_canvas.configure(yscrollcommand=secondary_scrollbar.set)

    secondary_attributes_frame = ttk.Frame(secondary_canvas)

    secondary_canvas.create_window((0, 0), window=secondary_attributes_frame, anchor="nw")
    secondary_scrollbar.pack(side="right", fill="y")
    secondary_canvas.pack(side="left", fill="both", expand=True)

    bind_mousewheel(secondary_canvas)

    secondary_check_vars = {}

    edit_attr_clash_file_button = ttk.Button(
        clash_window, text="Edit Attribute Clash File", command=lambda: edit_file_content(ATTRIBUTE_CLASH_FILE)
    )
    edit_attr_clash_file_button.pack(side="bottom", pady=5)

    save_button = ttk.Button(clash_window, text="Save Clashes", command=save_attribute_clashes)
    save_button.pack(side="bottom", pady=5)


# Main application window
def main_app():
    def open_selected_file():
        selected_index = file_listbox.curselection()
        if selected_index:
            selected_file = file_listbox.get(selected_index)
            edit_file(os.path.join(TXT_DIRECTORY, selected_file))

    root.title("Prompt Attribute Probability Editor")
    root.geometry("600x500")

    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)

    scrollbar = ttk.Scrollbar(frame, orient=ttk.VERTICAL)
    scrollbar.pack(side="right", fill="y")

    global file_listbox
    file_listbox = ttk.Listbox(frame, yscrollcommand=scrollbar.set)
    file_listbox.pack(fill="both", expand=True)

    for file in files:
        file_listbox.insert(ttk.END, file)

    scrollbar.config(command=file_listbox.yview)

    global file_context_menu

    file_context_menu = ttk.Menu(root, tearoff=0)
    file_context_menu.add_command(label="Create New File", command=create_file)
    file_context_menu.add_command(
        label="Edit File",
        command=lambda: edit_file_content(os.path.join(TXT_DIRECTORY, file_listbox.get(file_listbox.curselection()))),
    )
    file_context_menu.add_command(label="Delete File", command=delete_file)
    file_context_menu.add_command(label="Edit Attribute Probabilities", command=open_selected_file)

    root.bind("<Button-3>", show_context_menu)

    edit_prob_button = ttk.Button(root, text="Edit Probability for Each Text File", command=edit_file_probabilities)
    edit_prob_button.pack(pady=5)

    declare_clash_button = ttk.Button(root, text="Declare Attribute Clashes", command=declare_attribute_clashes)
    declare_clash_button.pack(pady=5)


# Start the application
if __name__ == "__main__":
    root = ttk.Tk()

    # Get list of text files
    files = [file for file in os.listdir(TXT_DIRECTORY) if file.endswith(".txt")]
    file_probabilities = {k: v for k, v in file_probabilities.items() if k in files}

    # Launch the main app
    main_app()
    root.mainloop()
