binary = [0,1,0,1,1,0]
rotations = [binary[i:] + binary[:i] for i in range(len(binary))]
print(rotations)
print(min(rotations))