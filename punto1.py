import numpy as np
import torch

# 1. Se crea un tensor x de tamaño 3x2 con valores aleatorios entre 0 y 1.
print("1. Creando tensor 'x' (3x2) con valores aleatorios:")
x = torch.rand(3, 2)
print("Tensor x:")
print(x)

# 2. Se crea un tensor y del mismo tamaño que x, pero lleno de unos.
print("\n2. Creando tensor 'y' (3x2) lleno de unos:")
y = torch.ones(x.size())
print("Tensor y:")
print(y)

# 3. Se suman los tensores x e y de forma elemento a elemento.
print("\n3. Sumando 'x' e 'y' para obtener 'z':")
z = x + y
print("Tensor z = x + y:")
print(z)

# 4. Se realizan operaciones de suma:
#    - z.add(1) crea un nuevo tensor sin modificar z.
#    - z.add_(1) modifica z en el lugar (in-place), sumándole 1 a cada elemento.
print("\n4. Sumando 1 a cada elemento de 'z':")
new_z = z.add(1)  # No modifica z, solo crea new_z
z.add_(1)         # Modifica z in-place
print("Tensor z después de aplicar z.add_(1):")
print(z)

# 5. Se imprime el tamaño del tensor z.
print("\n5. Tamaño del tensor z:")
print(z.size())

# 6. Se cambia la forma del tensor z de (3,2) a (2,3) utilizando resize_ (operación in-place).
print("\n6. Cambiando la forma de 'z' a (2,3) con resize_:")
z.resize_(2, 3)
print("Tensor z después de resize_:")
print(z)

print("\n------------ Conversión entre NumPy y Torch -----------------")

# 7. Se crea un array de NumPy 'a' de tamaño 4x3 con valores aleatorios.
print("\n7. Creando array de NumPy 'a' (4x3) con valores aleatorios:")
a = np.random.rand(4, 3)
print("Array a:")
print(a)

# 8. Se convierte el array de NumPy a un tensor de PyTorch 'b'.
print("\n8. Convirtiendo 'a' a tensor de PyTorch 'b':")
b = torch.from_numpy(a)
print("Tensor b (desde a):")
print(b)

# 9. Se convierte 'b' nuevamente a un array de NumPy.
b_new = b.numpy()
print("\n9. Convirtiendo 'b' nuevamente a array de NumPy (b_new):")
print(b_new)

# 10. Se multiplica in-place el tensor 'b' por 2.
print("\n10. Multiplicando 'b' in-place por 2:")
b.mul_(2)
print("Tensor b después de b.mul_(2):")
print(b)

# 11. Dado que 'b' y 'a' comparten la misma memoria, al modificar 'b' también se actualiza 'a'.
print("\n11. Como 'b' y 'a' comparten memoria, 'a' también se actualiza:")
print("Array a actualizado:")
print(a)
