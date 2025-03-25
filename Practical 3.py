import tkinter as tk
from tkinter import ttk

# Function to classify even (1) and odd (0)
def classify_number(n):
    return 1 if n % 2 == 0 else 0  # Even -> 1, Odd -> 0

# Create main window
root = tk.Tk()
root.title("Perceptron: Even/Odd Classification")
root.geometry("400x350")
root.configure(bg="#2C3E50")  # Dark background

# Style for the Treeview table
style = ttk.Style()
style.theme_use("clam")  # Use a clean style
style.configure("Treeview", 
                background="#ECF0F1", 
                foreground="black", 
                rowheight=30, 
                fieldbackground="#ECF0F1")
style.configure("Treeview.Heading", 
                font=("Arial", 12, "bold"), 
                background="#2980B9", 
                foreground="white")

# Create table using Treeview
columns = ("Number", "ASCII Value", "Even (1) / Odd (0)")
tree = ttk.Treeview(root, columns=columns, show="headings", height=10)
tree.heading("Number", text="Number")
tree.heading("ASCII Value", text="ASCII Value")
tree.heading("Even (1) / Odd (0)", text="Even (1) / Odd (0)")

# Center columns and set width
for col in columns:
    tree.column(col, anchor="center", width=120)

# Insert data into the table
for num in range(10):  # ASCII digits 0-9
    ascii_val = ord(str(num))
    even_odd = classify_number(num)
    tree.insert("", "end", values=(num, ascii_val, even_odd))

tree.pack(pady=20, padx=20)

# Styled Exit button
exit_button = tk.Button(root, text="Exit", command=root.quit, 
                        bg="#E74C3C", fg="white", font=("Arial", 12, "bold"), 
                        padx=10, pady=5, relief="flat", activebackground="#C0392B")

exit_button.pack(pady=10)

# Run application
root.mainloop()
