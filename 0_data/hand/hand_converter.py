import trimesh
import numpy as np

hand = trimesh.load_mesh("hand_only_simplified3k5.stl")
data = np.array(hand.vertices)

np.save("hand_array3k5.npy", data)
