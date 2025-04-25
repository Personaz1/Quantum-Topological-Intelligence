import matplotlib.pyplot as plt

def plot_memory(memory_state, title='Memory State'):
    plt.imshow(memory_state, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()
