import numpy as np
from ripser import ripser
from persim import plot_diagrams
import gudhi as gd

class Memory:
    """
    Топологическая память для Difference Loop.
    Хранит состояние (матрица), историю различий (point cloud) и позволяет вычислять persistent homology.

    Пример использования:
        mem = Memory(shape=(10, 10))
        mem.deform(np.array([1.0, 2.0]))
        dgms = mem.persistent_homology()
        mem.plot_phases()
    """
    def __init__(self, shape=(10, 10)):
        """
        Инициализация памяти.
        :param shape: tuple, форма внутренней матрицы памяти
        """
        self.shape = shape
        self.state = np.zeros(shape)
        self.history = []  # Храним историю деформаций (различий)

    def deform(self, diff_vector):
        """
        Деформирует память на основе входного различия.
        Простейшая реализация: добавляет норму diff_vector в случайную точку поверхности.
        :param diff_vector: np.ndarray, вектор различий
        """
        idx = tuple(np.random.randint(0, s) for s in self.shape)
        self.state[idx] += np.linalg.norm(diff_vector)
        self.history.append(diff_vector.copy())

    def get_state(self):
        """
        Возвращает текущее состояние памяти (матрица).
        :return: np.ndarray
        """
        return self.state

    def get_history(self):
        """
        Возвращает накопленные различия (point cloud для TDA).
        :return: np.ndarray, shape=(steps, input_dim)
        """
        return np.array(self.history)

    def persistent_homology(self, method="ripser"):
        """
        Вычисляет persistent homology по истории различий.
        :param method: str, 'ripser' или 'gudhi'
        :return: persistence diagrams (list of np.ndarray)
        """
        points = self.get_history()
        if len(points) < 2:
            return None  # Недостаточно данных
        if method == "ripser":
            result = ripser(points)
            return result['dgms']
        elif method == "gudhi":
            rips_complex = gd.RipsComplex(points=points, max_edge_length=2.0)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            diag = simplex_tree.persistence()
            return simplex_tree.persistence_intervals_in_dimension(1)
        else:
            raise ValueError("Unknown method")

    def plot_persistence(self, method="ripser"):
        """
        Визуализирует persistence diagrams.
        :param method: str, 'ripser' или 'gudhi'
        """
        dgms = self.persistent_homology(method=method)
        if dgms is None:
            print("Недостаточно данных для persistent homology")
            return
        if method == "ripser":
            plot_diagrams(dgms, show=True)
        elif method == "gudhi":
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("Persistence intervals (H1)")
            plt.xlabel("Birth")
            plt.ylabel("Death")
            intervals = dgms
            for birth, death in intervals:
                plt.plot([birth, death], [1, 1], 'b')
            plt.show()

    def plot_phases(self, method="ripser"):
        """
        Визуализирует динамику памяти и фазовые переходы:
        - График нормы состояния памяти
        - Динамика числа компонент (H0) и циклов (H1) persistent homology по времени
        :param method: str, 'ripser' или 'gudhi'
        """
        import matplotlib.pyplot as plt
        norms = []
        h0_counts = []
        h1_counts = []
        # Для каждой точки истории считаем PH по накопленным данным
        for i in range(2, len(self.history)+1):
            points = np.array(self.history[:i])
            norms.append(np.linalg.norm(self.state))
            if method == "ripser":
                from ripser import ripser
                dgms = ripser(points)['dgms']
                h0_counts.append(len(dgms[0]))
                h1_counts.append(len(dgms[1]) if len(dgms) > 1 else 0)
            elif method == "gudhi":
                import gudhi as gd
                rips_complex = gd.RipsComplex(points=points, max_edge_length=2.0)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
                diag = simplex_tree.persistence()
                h0 = [d for d in diag if d[0] == 0]
                h1 = [d for d in diag if d[0] == 1]
                h0_counts.append(len(h0))
                h1_counts.append(len(h1))
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        axs[0].plot(norms, label='||state||')
        axs[0].set_ylabel('Norm of Memory')
        axs[0].legend()
        axs[1].plot(h0_counts, label='H0 (components)')
        axs[1].set_ylabel('H0 count')
        axs[1].legend()
        axs[2].plot(h1_counts, label='H1 (cycles)')
        axs[2].set_ylabel('H1 count')
        axs[2].set_xlabel('Step')
        axs[2].legend()
        plt.suptitle('Динамика памяти и фазовые переходы')
        plt.show()
