import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import matplotlib as mpl

# 设置SCI论文风格的全局参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'

class VoxelStructure3D:
    def __init__(self, grid_size=10, voxel_size=0.2):
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.total_size = grid_size * voxel_size
        
        # Initialize voxel grid
        self.voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)
        
        # Color mapping - lighter for annotation
        self.color_map = {
            'beam_x': '#FFB3B3',
            'beam_y': '#B3FFB3',
            'column': '#B3D9FF',
            'node': '#FFE699',
            'empty': '#F5F5F5'
        }
        
        self.column_size_xy = 3
        self.node_height = 4
        
    def add_column_full_height(self, x, y):
        start_x = x - self.column_size_xy // 2
        start_y = y - self.column_size_xy // 2
        
        for i in range(self.column_size_xy):
            for j in range(self.column_size_xy):
                for k in range(self.grid_size):
                    px = start_x + i
                    py = start_y + j
                    pz = k
                    if 0 <= px < self.grid_size and 0 <= py < self.grid_size and 0 <= pz < self.grid_size:
                        self.voxel_grid[px, py, pz] = 2
    
    def add_node_beam_xy(self, x, y, z, direction='one_face'):
        # Add column
        self.add_column_full_height(x, y)
        
        # Add node
        start_x = x - self.column_size_xy // 2
        start_y = y - self.column_size_xy // 2
        start_z = z - self.node_height // 2
        
        for i in range(self.column_size_xy):
            for j in range(self.column_size_xy):
                for k in range(self.node_height):
                    px = start_x + i
                    py = start_y + j
                    pz = start_z + k
                    if 0 <= px < self.grid_size and 0 <= py < self.grid_size and 0 <= pz < self.grid_size:
                        self.voxel_grid[px, py, pz] = 5
        
        # X-beam (+X direction)
        beam_x_start = x + self.column_size_xy // 2
        beam_y = y
        beam_z = z
        
        start_y = beam_y - self.column_size_xy // 2
        start_z = beam_z - self.node_height // 2
        
        length_x = self.grid_size - beam_x_start
        for i in range(self.column_size_xy):
            for j in range(self.node_height):
                for k in range(length_x):
                    px = beam_x_start + k
                    py = start_y + i
                    pz = start_z + j
                    if 0 <= px < self.grid_size and 0 <= py < self.grid_size and 0 <= pz < self.grid_size:
                        if self.voxel_grid[px, py, pz] != 5:
                            self.voxel_grid[px, py, pz] = 3
        
        # Y-beam (+Y direction)
        beam_x = x
        beam_y_start = y + self.column_size_xy // 2
        beam_z = z
        
        start_x = beam_x - self.column_size_xy // 2
        start_z = beam_z - self.node_height // 2
        
        length_y = self.grid_size - beam_y_start
        for i in range(self.column_size_xy):
            for j in range(self.node_height):
                for k in range(length_y):
                    px = start_x + i
                    py = beam_y_start + k
                    pz = start_z + j
                    if 0 <= px < self.grid_size and 0 <= py < self.grid_size and 0 <= pz < self.grid_size:
                        if self.voxel_grid[px, py, pz] != 5:
                            self.voxel_grid[px, py, pz] = 4
        
        if direction == 'two_faces':
            # X-beam (-X direction)
            beam_x_end = x - self.column_size_xy // 2
            for i in range(self.column_size_xy):
                for j in range(self.node_height):
                    for k in range(beam_x_end):
                        px = k
                        py = start_y + i
                        pz = start_z + j
                        if 0 <= px < self.grid_size and 0 <= py < self.grid_size and 0 <= pz < self.grid_size:
                            if self.voxel_grid[px, py, pz] != 5:
                                self.voxel_grid[px, py, pz] = 3
            
            # Y-beam (-Y direction)
            beam_y_end = y - self.column_size_xy // 2
            for i in range(self.column_size_xy):
                for j in range(self.node_height):
                    for k in range(beam_y_end):
                        px = start_x + i
                        py = k
                        pz = start_z + j
                        if 0 <= px < self.grid_size and 0 <= py < self.grid_size and 0 <= pz < self.grid_size:
                            if self.voxel_grid[px, py, pz] != 5:
                                self.voxel_grid[px, py, pz] = 4
    
    def clear(self):
        self.voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=int)
    
    def get_color_grid(self):
        color_grid = np.empty((self.grid_size, self.grid_size, self.grid_size), dtype=object)
        color_grid.fill(self.color_map['empty'])
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    val = self.voxel_grid[x, y, z]
                    if val == 2:
                        color_grid[x, y, z] = self.color_map['column']
                    elif val == 3:
                        color_grid[x, y, z] = self.color_map['beam_x']
                    elif val == 4:
                        color_grid[x, y, z] = self.color_map['beam_y']
                    elif val == 5:
                        color_grid[x, y, z] = self.color_map['node']
        
        return color_grid
    
    def plot_clean(self):
        """Plot clean structure without annotations"""
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create structure
        self.clear()
        self.add_node_beam_xy(5, 5, 5, 'one_face')
        
        # Get filled and color grid
        filled = self.voxel_grid > 0
        color_grid = self.get_color_grid()
        
        # Plot voxels with lighter colors
        ax.voxels(
            filled,
            facecolors=color_grid,
            edgecolor='gray',
            alpha=0.6,
            linewidth=0.2
        )
        
        # Set axis labels
        ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
        
        # Set axis limits
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_zlim(0, self.grid_size)
        
        # Set tick positions
        tick_positions = np.arange(0, self.grid_size+1, 2)
        tick_labels = [f'{i*self.voxel_size:.1f}' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_zticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_yticklabels(tick_labels, fontsize=10)
        ax.set_zticklabels(tick_labels, fontsize=10)
        
        # Set view angle
        ax.view_init(elev=25, azim=-55)
        ax.grid(True, alpha=0.1, linewidth=0.2)
        
        # Title
        ax.set_title('(h) Node+XY-beams (1-face)', 
                    fontsize=12, fontweight='bold', pad=20)
        
        # No legend
        plt.tight_layout()
        
        return fig, ax

def main():
    """Main function - generate clean figure"""
    model = VoxelStructure3D(grid_size=10, voxel_size=0.2)
    
    # Generate clean figure
    fig, ax = model.plot_clean()
    
    # Save figures
    print("Saving clean structural figure...")
    
    fig.savefig('structural_clean.pdf', 
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.1,
                format='pdf')
    print("✓ PDF saved: structural_clean.pdf")
    
    fig.savefig('structural_clean.png', 
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.1,
                format='png')
    print("✓ PNG saved: structural_clean.png")
    
    plt.show()
    
    return model, fig

if __name__ == "__main__":
    model, fig = main()