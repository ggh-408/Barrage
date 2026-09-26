"""Independent-process focus target for window scheduling diagnostics."""
import tkinter as tk

root = tk.Tk()
root.title('Barrage focus control')
root.geometry('280x90+20+40')
tk.Label(root, text='Independent focus target\nGame runs in a separate process').pack(padx=10, pady=15)
root.mainloop()
