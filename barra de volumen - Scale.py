import tkinter as tk

def update_value(val):
    # Actualiza el valor con dos decimales
    value_label.config(text=f"{float(val):.2f}")

root = tk.Tk()

# Crea una escala vertical con rango de 0 a 1, incrementos de 0.01
scale = tk.Scale(root, from_=1, to=0, resolution=0.01, orient='vertical', command=update_value)
scale.pack()

# Etiqueta para mostrar el valor actualizado
value_label = tk.Label(root, text="0.00")
value_label.pack()

root.mainloop()
