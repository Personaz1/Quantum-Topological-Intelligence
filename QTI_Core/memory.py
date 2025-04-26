import numpy as np
from ripser import ripser
from persim import plot_diagrams
import gudhi as gd

class Memory:
    """
    Topological memory for the Difference Loop.
    Stores the state (matrix), history of differences (point cloud), and allows computing persistent homology.

    Example usage:
        mem = Memory(shape=(10, 10))
        mem.deform(np.array([1.0, 2.0]))
        state = mem.retrieve_state()
        history = mem.get_history()
        persistence = mem.compute_persistent_homology()
        mem.visualize_dynamics()
        mem.visualize_persistent_homology(persistence)
        transition = mem.detect_phase_transition()
        print(f"Detected phase transition: {transition}")
    """
    def __init__(self, shape=(10, 10)):
        """
        Initializes the memory.
        :param shape: tuple, shape of the internal memory matrix
        """
        self.shape = shape
        self.state = np.zeros(shape)
        self.history = []  # Stores the history of deformations (differences)
        self.point_cloud = np.array([])

    def deform(self, difference: np.ndarray) -> None:
        """
        Deforms the memory state based on the difference vector and adds the difference to the history.
        Simplest implementation: adds the norm of difference to a random point on the surface.
        :param difference: np.ndarray, difference vector
        """
        idx = tuple(np.random.randint(0, s) for s in self.shape)
        self.state[idx] += np.linalg.norm(difference)
        self.history.append(difference.copy())
        self.point_cloud = np.append(self.point_cloud, difference.copy())

    def retrieve_state(self) -> np.ndarray:
        """
        Retrieves the current state of the memory matrix.
        :return: np.ndarray
        """
        return self.state

    def get_state(self) -> np.ndarray:
        """
        Alias for retrieve_state (for compatibility).
        :return: np.ndarray
        """
        return self.retrieve_state()

    def get_history(self) -> list:
        """
        Retrieves the history of difference vectors.
        :return: list of np.ndarray, shape=(steps, input_dim)
        """
        return self.history

    def compute_persistent_homology(self, method="ripser", use_point_cloud=False):
        """
        Computes persistent homology based on the history of differences or the point cloud.
        :param method: str, 'ripser' or 'gudhi'
        :param use_point_cloud: bool, if True, use self.point_cloud, otherwise self.history
        :return: persistence diagrams (list of np.ndarray)
        """
        points = self.point_cloud if use_point_cloud else self.history
        if len(points) < 2:
            return None  # Not enough data
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

    def visualize_dynamics(self) -> None:
        """
        Visualizes the dynamics of memory changes over time.
        """
        import matplotlib.pyplot as plt
        norms = []
        h0_counts = []
        h1_counts = []
        # For each point in history, compute PH on accumulated data
        for i in range(2, len(self.history)+1):
            points = np.array(self.history[:i])
            norms.append(np.linalg.norm(self.state))
            dgms = self.compute_persistent_homology(method="ripser")
            h0_counts.append(len(dgms[0]))
            h1_counts.append(len(dgms[1]) if len(dgms) > 1 else 0)
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
        plt.suptitle('Memory Dynamics and Phase Transitions')
        plt.show()

    def detect_phase_transition(self) -> bool:
        """
        Detects phase transitions in the memory dynamics.
        :return: bool, True if a phase transition is detected, False otherwise
        """
        # Implementation of phase transition detection logic
        return False  # Placeholder return, actual implementation needed

    def visualize_persistent_homology(self, persistence_diagrams: list) -> None:
        """
        Visualizes the computed persistent homology diagrams.
        :param persistence_diagrams: list of np.ndarray, persistence diagrams to visualize.
        """
        if persistence_diagrams is None:
            print("Not enough data for persistent homology")
            return
        if len(persistence_diagrams) > 0:
            plot_diagrams(persistence_diagrams, show=True)
        else:
            print("No persistent homology data to visualize")

    def add_point(self, point):
        """
        Adds a point to the point cloud (e.g., from an external source).
        :param point: np.ndarray
        """
        self.point_cloud = np.append(self.point_cloud, point.copy())

    def reset_point_cloud(self):
        """
        Clears the point cloud.
        """
        self.point_cloud = np.array([])

    def plot_persistence(self, method="ripser"):
        """
        Visualizes persistence diagrams.
        :param method: str, 'ripser' or 'gudhi'
        """
        dgms = self.compute_persistent_homology(method=method)
        if dgms is None:
            print("Not enough data for persistent homology")
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
        Visualizes memory dynamics and phase transitions:
        - Graph of memory state norm
        - Dynamics of the number of components (H0) and cycles (H1) of persistent homology over time
        :param method: str, 'ripser' or 'gudhi'
        """
        import matplotlib.pyplot as plt
        norms = []
        h0_counts = []
        h1_counts = []
        # For each point in history, compute PH on accumulated data
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
        plt.suptitle('Memory Dynamics and Phase Transitions')
        plt.show()
