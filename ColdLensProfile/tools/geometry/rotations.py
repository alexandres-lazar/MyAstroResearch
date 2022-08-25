#! usr/bin/env python3

import random
import numpy as np

# ---------------------------------------------------------------------------

def main() -> None:
    npoints = 200
    pts = 2500 * np.random.random((npoints, 3))
    print(CoordinateRotation3D(pts))

# ---------------------------------------------------------------------------

class CoordinateRotation3D(object):
        
    def __init__(self, coords):
        if isinstance(coords, np.ndarray):
            self.coords = coords
        else:
            raise Exception("!!! Coordinate must be a three dimensional vector numpy array!!")

    def __len__(self) -> int:
        return len(self.coords)

    def uniform_rotation(self) -> float:
        """Coordinate transformation using uniform matrix"""
        result = np.zeros(self.coords.shape)
        for ind, coord in enumerate(self.coords):
            result[ind] = np.inner(self._uniform_matrix(), coord.T)
        return result 

    def random_rotation(self, seed: int = None) -> float:
        """Coordinate transformation using randomly generated rotation matrix"""
        transform = np.inner(self._randomized_rotation_matrix(seed), self.coords[:])
        return np.stack(transform, axis=-1)

    def angle_rotation(self, psi: float, theta: float, phi: float) -> float:
        """Cartesian coordinate transformation based on given spherical rotation angles"""
        rot_mat = self+rotation_matrix_3D(psi, theta, phi) 
        transform = np.inner(rot_mat, self_coords[:])
        return np.stack(transform, axis=-1)
    
    def rotation_new_z_axis(self, new_unit_vector: float = [0.0, 0.0, 1.0]) -> float:
        """Rotation transformation of coordinates `coords` to align 
           with a specifed `unit_vector`
        """
        matrix = self._rotation_matrix_along_unit_vector(new_unit_vector)
        transform = np.inner(matrix, self.coords[:])
        return np.stack(transform, axis=-1)
    
    def _uniform_matrix(self) -> float:
        """Constructs a 3x3 identity matric"""
        matrix = np.array(
                [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]
                )
        return matrix

    def _randomized_rotation_matrix(self, seed: int = None) -> float:
        """Randomly generated rotation matrix"""
        
        if isinstance(seed, int):
            random.seed(seed)
        else:
            pass
        
        # Sample angles from uniform distribution
        r_psi = random.uniform(0.0, 2.0*np.pi)
        r_theta = random.uniform(0.0, 2.0*np.pi)   
        r_phi = random.uniform(0.0, 2.0*np.pi)
        
        return self._rotation_matrix(r_psi, r_theta, r_phi)

    def _rotation_matrix(self, psi: float, theta: float, phi: float) -> float:
        """3D rotation matrix given spherical coordinates
             R_zyz rotation -> R_z(phi)R_y(theta)R_z(psi)
        """ 
        # diagonal components
        u00 = np.cos(phi)*np.cos(theta)*np.cos(psi) - np.sin(phi)*np.sin(psi)
        u11 = -np.sin(phi)*np.cos(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi)
        u22 = np.cos(theta)
        
        # off-diagonal components
        u01 = -np.cos(phi)*np.cos(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)
        u02 = np.cos(phi) * np.sin(theta)
        u10 = np.sin(phi)*np.cos(theta)*np.cos(psi) + np.cos(phi)*np.sin(psi)
        u12 = np.sin(phi) * np.sin(theta)
        u20 = - np.sin(theta) * np.cos(psi)
        u21 = np.sin(theta) * np.sin(psi)

        matrix = np.array(
                [[u00, u01, u02],
                 [u10, u11, u12],
                 [u20, u21, u22]]
                )
        return matrix

    def _rotation_matrix_along_unit_vector(self, 
                                           new_unit_vector: float, 
                                           unit_vector_z: float = [0.0, 0.0, 1.0]
                                           ) -> float:
        """Returns a rotation matrix that, when applied to coordinates, transforms
           the coordinates to have the z-coordinates align along `new_unit_vector`
        Applied method discussed in: https://stackoverflow.com/questions/43507491/ 
        imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis
        """
        unit = np.array(unit_vector_z)
        # normalize unit vector and compute dot and cross product with unit vector
        new_unit = new_unit_vector / np.linalg.norm(new_unit_vector)
        uvw = np.cross(new_unit, unit)
        rcos = np.dot(new_unit, unit)
        rsin = np.linalg.norm(uvw);
        
        if not np.isclose(rsin, 0):
            uvw /= rsin
            u, v, w = uvw
        
        matrix = np.array(
                        [[0.0, -w, v],
                         [w, 0.0, -u],
                         [-v, u, 0.0]]
                        )
        return rcos*np.eye(3) + rsin*matrix + (1.0 - rcos)*uvw[:,None]*uvw[None,:]


if __name__ == "__main__":
    main()
