import os

# Ruta al directorio que contiene las carpetas
directory = '/home/pixelguy14/Documentos/SemAgo-Dic-2024/Topicos_Sistemas_Comp/Proyecto_Final/clean_dataset'

# Lista para almacenar los nombres de las carpetas
folder_names = []

# Recorrer las carpetas en el directorio
for folder_name in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, folder_name)):
        folder_names.append(folder_name)

# Ordenar los nombres de las carpetas alfabéticamente
folder_names.sort()

# Formatear los nombres de las carpetas en el estilo solicitado
formatted_folder_names = [f"'{name}'" for name in folder_names]

# Contar el número de carpetas
folder_count = len(formatted_folder_names)

# Guardar los nombres de las carpetas en un archivo de texto
output_file = 'folder_names.txt'
with open(output_file, 'w') as file:
    file.write(", ".join(formatted_folder_names) + '\n')
    file.write(f"Total de carpetas: {folder_count}\n")

print(f"Nombres de las carpetas guardados en '{output_file}'")
print(f"Total de carpetas: {folder_count}")
