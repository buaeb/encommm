"""Explicit-membrane ENCoM (exENCoM).

Embeds the protein in an elastic lattice representing the lipid bilayer and integrates the membrane out with the
vibrational subsystem analysis (VSA), so that the normal modes of the protein feel the lateral constraints imposed by
the membrane. The membrane is split into concentric rings around the protein and the reduced Hessian is computed with
the recursive Schur complement formula (matrix continued fraction) of Zhang, Zhang, Xing and Bahar 2021:
`doi.org/10.1063/5.0046710 <https://doi.org/10.1063/5.0046710>`_. The membrane lattice follows the exANM model of
Lezon and Bahar 2012: `doi.org/10.1016/j.bpj.2012.02.028 <https://doi.org/10.1016/j.bpj.2012.02.028>`_

The protein (system) Hessian is the regular ENCoM Hessian. Membrane-membrane and protein-membrane interactions are
ANM-style springs, as in the paper.

The membrane lattice construction and the ring decomposition are adapted from ProDy (prody/dynamics/exanm.py),
which is distributed under the MIT License:

    ProDy: A Python Package for Protein Dynamics Analysis

    Copyright (C) 2010-2014 University of Pittsburgh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
"""
import numpy as np
from nrgten.encom import ENCoM


class ExENCoM(ENCoM):
    """ENCoM with an explicit membrane integrated out by vibrational subsystem analysis.

    The PDB file must already be oriented like the structures from the OPM database or the PPM server, i.e. with the
    membrane normal along z and the membrane center at z = 0. The resulting Hessian has the same size as the regular
    ENCoM Hessian (only the protein masses remain), so every method inherited from ENM works unchanged. The reduced
    Hessian keeps the 6 rigid-body zero modes of the protein-membrane system.

    Note:
        ExENCoM extends the `ENCoM` class, to see inherited attributes look at the documentation for `ENCoM` and `ENM`.

    Attributes:
        memb_h (float): Half thickness of the membrane (Å). The membrane spans -memb_h < z < memb_h.
        memb_R (float): Outer radius of the membrane in the x-y plane (Å).
        memb_Ri (float): Inner radius of the membrane (Å), 0 means not hollow.
        node_r (float): Radius of each membrane node (Å). Nearest neighbors are 2 * node_r apart.
        lattice (str): Lattice type, "FCC" (face-centered cubic), "SC" (simple cubic) or "SH" (simple hexagonal).
        exr (float): Membrane nodes closer than this to a transmembrane protein mass are removed (Å).
        use_hull (bool): If True, membrane nodes inside the convex hull of the transmembrane part of the protein are
                         removed (requires scipy). Turn off for multimers with a central pore filled by lipids.
        memb_cut (float): Cutoff distance (Å) of the membrane-membrane and protein-membrane springs.
        memb_k (float): Spring constant of the membrane springs. If None, it is set to memb_stiffness times the
                        ANM spring constant that matches the trace of the ENCoM Hessian (see match_spring_constant).
        memb_stiffness (float): Membrane stiffness relative to the protein, used only when memb_k is None.
        ring_r0 (float): Width (Å) of the first membrane ring around the protein (must be >= memb_cut).
        ring_dr (float): Width (Å) of the outer membrane rings (must be >= memb_cut).
        membrane (numpy.ndarray): Coordinates of the membrane nodes (n x 3), set when the Hessian is built.
        memb_k_used (float): The spring constant actually used for the membrane.
    """

    def __init__(self, pdb_file, membrane_half_thickness=13., membrane_radius=80., membrane_inner_radius=0.,
                 node_radius=3.1, lattice="FCC", exclusion_radius=5., use_hull=True, membrane_cutoff=15.,
                 membrane_k=None, membrane_stiffness=1., ring_r0=20., ring_dr=20., **kwargs):
        """Constructor for the ExENCoM class.

        Args:
            pdb_file (str): The PDB file to read, oriented as in OPM/PPM (membrane normal along z, centered at z=0).
            membrane_half_thickness (float, optional): Half thickness of the membrane (Å).
            membrane_radius (float, optional): Outer radius of the membrane (Å).
            membrane_inner_radius (float, optional): Inner radius of the membrane (Å), 0 means not hollow.
            node_radius (float, optional): Radius of each membrane node (Å).
            lattice (str, optional): "FCC", "SC" or "SH".
            exclusion_radius (float, optional): Exclusion radius around transmembrane protein masses (Å).
            use_hull (bool, optional): If True, removes membrane nodes inside the protein's convex hull.
            membrane_cutoff (float, optional): Cutoff of the membrane springs (Å).
            membrane_k (float, optional): Membrane spring constant. If None, it is matched to the ENCoM Hessian.
            membrane_stiffness (float, optional): Membrane stiffness relative to the protein when membrane_k is None.
            ring_r0 (float, optional): Width of the first membrane ring (Å).
            ring_dr (float, optional): Width of the outer membrane rings (Å).
            **kwargs: Passed to the ENCoM constructor (kr, epsi, use_pickle, solve, ...).
        """
        if kwargs.get("massWeighted", False):
            raise ValueError("ExENCoM does not support massWeighted=True")
        if ring_r0 < membrane_cutoff or ring_dr < membrane_cutoff:
            raise ValueError("ring_r0 and ring_dr must be >= membrane_cutoff so that only neighboring rings interact")
        self.memb_h = float(membrane_half_thickness)
        self.memb_R = float(membrane_radius)
        self.memb_Ri = float(membrane_inner_radius)
        self.node_r = float(node_radius)
        self.lattice = lattice
        self.exr = float(exclusion_radius)
        self.use_hull = use_hull
        self.memb_cut = float(membrane_cutoff)
        self.memb_k = membrane_k
        self.memb_stiffness = float(membrane_stiffness)
        self.ring_r0 = float(ring_r0)
        self.ring_dr = float(ring_dr)
        self.membrane = None
        self.memb_k_used = None
        super().__init__(pdb_file, **kwargs)

    def build_hessian(self):
        """Builds the reduced Hessian of the protein embedded in the membrane.

        Returns:
            The reduced Hessian matrix (3n x 3n, n being the number of protein masses).
        """
        h_prot = super().build_hessian()
        # the per-term Hessians are not needed anymore and take 4 times the memory of the Hessian
        self.V1_H, self.V2_H, self.V3_H, self.V4_H = None, None, None, None
        prot_xyz = self._get_mass_xyz()
        self.membrane = build_membrane(prot_xyz, h=self.memb_h, R=self.memb_R, Ri=self.memb_Ri, r=self.node_r,
                                       lat=self.lattice, exr=self.exr, use_hull=self.use_hull)
        if self.memb_k is None:
            self.memb_k_used = self.memb_stiffness * match_spring_constant(h_prot, prot_xyz, self.memb_cut)
        else:
            self.memb_k_used = float(self.memb_k)
        self._print_verbose("exENCoM: {0} membrane nodes, membrane k = {1}".format(len(self.membrane),
                                                                                  self.memb_k_used))
        if len(self.membrane) == 0:
            raise ValueError("No membrane node was built. The PDB file should be oriented as in OPM/PPM.")
        return reduce_membrane_hessian(h_prot, prot_xyz, self.membrane, cutoff=self.memb_cut, k=self.memb_k_used,
                                       r0=self.ring_r0, dr=self.ring_dr, overwrite=True)

    def write_membrane(self, filename):
        """Writes the membrane nodes as HETATM records (residue NE1, chain Q) for visualization.

        Args:
            filename (str): The output PDB file.
        """
        assert self.membrane is not None
        with open(filename, 'w') as fh:
            for i, (x, y, z) in enumerate(self.membrane):
                fh.write("HETATM{0:>5d}  Q1  NE1 Q{1:>4d}    {2:8.3f}{3:8.3f}{4:8.3f}  1.00  0.00           Q\n".format(
                    (i + 1) % 100000, (i + 1) % 10000, x, y, z))
            fh.write("END\n")

    def _sum_hessians(self):
        # sums in place into V1_H instead of allocating temporaries (the terms are discarded in build_hessian)
        h = self.V1_H
        h += self.V2_H
        h += self.V3_H
        h += self.V4_H
        return h

    def _get_mass_xyz(self):
        return np.array([m[:3] for m in self.mol.masses], dtype=float)

    def _get_pickle_file(self):
        param_string = "_".join(str(x) for x in (self.memb_h, self.memb_R, self.memb_Ri, self.node_r, self.lattice,
                                                 self.exr, self.use_hull, self.memb_cut, self.memb_k,
                                                 self.memb_stiffness, self.ring_r0, self.ring_dr))
        return super()._get_pickle_file().replace(".pickle", ".exencom_{0}.pickle".format(param_string))


def _lattice_vectors(lat):
    """Returns the 3 primitive vectors (one per row) of the given lattice type, with unit nearest-neighbor distance.
    """
    if lat == "FCC":
        s = 1. / np.sqrt(2)
        return np.array([[0., s, s], [s, 0., s], [s, s, 0.]])
    if lat == "SC":
        return np.eye(3)
    if lat == "SH":
        return np.array([[0.5, -np.sqrt(3) / 2, 0.], [0.5, np.sqrt(3) / 2, 0.], [0., 0., 1.]])
    raise ValueError("Unknown lattice type: {0}".format(lat))


def _xy_center(prot_xyz):
    c0 = prot_xyz.mean(axis=0)
    c0[2] = 0.
    return c0


def build_membrane(prot_xyz, h=13., R=80., Ri=0., r=3.1, lat="FCC", exr=5., use_hull=True):
    """Builds a cylindrical membrane lattice around the protein.

    The lattice is centered on the protein's x-y centroid and spans -h < z < h, Ri < sqrt(x^2 + y^2) < R. Nodes
    closer than exr to a transmembrane protein mass, or inside the convex hull of the transmembrane part of the
    protein (if use_hull), are removed.

    Args:
        prot_xyz (numpy.ndarray): Coordinates of the protein masses (n x 3).
        h (float, optional): Half thickness of the membrane (Å).
        R (float, optional): Outer radius of the membrane (Å).
        Ri (float, optional): Inner radius of the membrane (Å).
        r (float, optional): Radius of each membrane node (Å).
        lat (str, optional): "FCC", "SC" or "SH".
        exr (float, optional): Exclusion radius around transmembrane protein masses (Å).
        use_hull (bool, optional): If True, removes the nodes inside the protein's convex hull (requires scipy).

    Returns:
        numpy.ndarray: Coordinates of the membrane nodes (m x 3), in the frame of prot_xyz.
    """
    c0 = _xy_center(prot_xyz)
    centered = prot_xyz - c0
    transmembrane = centered[np.logical_and(centered[:, 2] < h, centered[:, 2] > -h)]
    if len(transmembrane) == 0:
        raise ValueError("No protein mass is inside the membrane. The PDB file should be oriented as in OPM/PPM.")

    V = _lattice_vectors(lat)
    maxes = [int(np.ceil((R + V[i, 2] * h) / r)) for i in range(3)]
    ijk = np.stack(np.meshgrid(*[np.arange(-m, m) for m in maxes], indexing='ij'), axis=-1).reshape(-1, 3)
    nodes = 2. * r * np.dot(ijk, V)
    dd = np.linalg.norm(nodes[:, :2], axis=1)
    keep = ((nodes[:, 2] > -h) & (nodes[:, 2] < h) & (np.abs(nodes[:, 0]) < R) & (np.abs(nodes[:, 1]) < R) &
            (dd < R) & (dd > Ri))
    nodes = nodes[keep]

    if use_hull:
        from scipy.spatial import ConvexHull
        eqs = ConvexHull(transmembrane).equations
        in_hull = np.all(np.dot(nodes, eqs[:, :3].T) + eqs[:, 3] <= 0, axis=1)
        nodes = nodes[~in_hull]

    clash = np.zeros(len(nodes), dtype=bool)
    exr_sq = exr ** 2
    for start in range(0, len(nodes), 1000):
        chunk = nodes[start:start + 1000]
        d2 = np.sum((chunk[:, None, :] - transmembrane[None, :, :]) ** 2, axis=2)
        clash[start:start + 1000] = np.any(d2 < exr_sq, axis=1)
    return nodes[~clash] + c0


def match_spring_constant(h_prot, prot_xyz, cutoff):
    """Returns the ANM spring constant k for which an ANM of the protein (with the given cutoff) has the same Hessian
    trace as h_prot, i.e. the average stiffness of the ENCoM springs expressed as an ANM spring.

    Args:
        h_prot (numpy.ndarray): The protein Hessian.
        prot_xyz (numpy.ndarray): Coordinates of the protein masses (n x 3).
        cutoff (float): The ANM cutoff (Å).

    Returns:
        float: The matching spring constant.
    """
    d2 = np.sum((prot_xyz[:, None, :] - prot_xyz[None, :, :]) ** 2, axis=2)
    n_pairs = (np.count_nonzero((d2 <= cutoff ** 2) & (d2 > 0))) // 2
    if n_pairs == 0:
        raise ValueError("No pair of protein masses within the membrane cutoff")
    # each ANM pair (2 * k * v v^T / |v|^2 on both diagonal blocks, see anm.py) adds 4k to the trace
    return np.trace(h_prot) / (4. * n_pairs)


def _spring_blocks(A, B, cutoff, k):
    """Returns the indices (ia, ib) and 3x3 blocks 2 * k * v v^T / |v|^2 of the ANM springs between A and B.
    """
    v = A[:, None, :] - B[None, :, :]
    d2 = np.sum(v ** 2, axis=2)
    ia, ib = np.nonzero((d2 <= cutoff ** 2) & (d2 > 0))
    vv = v[ia, ib]
    blocks = 2. * k * vv[:, :, None] * vv[:, None, :] / d2[ia, ib][:, None, None]
    return ia, ib, blocks


def _intra_hessian(X, cutoff, k):
    n = len(X)
    H = np.zeros((n, 3, n, 3))
    ia, ib, blocks = _spring_blocks(X, X, cutoff, k)
    H[ia, :, ib, :] = -blocks
    diag = np.zeros((n, 3, 3))
    np.add.at(diag, ia, blocks)
    H[np.arange(n), :, np.arange(n), :] += diag
    return H.reshape(3 * n, 3 * n)


def _coupling(A, B, cutoff, k, with_F=True):
    """Returns the off-diagonal coupling block F (3nA x 3nB) between A and B (None if not with_F) and the contribution
    of these springs to the diagonal 3x3 blocks of A (nA x 3 x 3).
    """
    nA, nB = len(A), len(B)
    ia, ib, blocks = _spring_blocks(A, B, cutoff, k)
    diag = np.zeros((nA, 3, 3))
    np.add.at(diag, ia, blocks)
    if not with_F:
        return None, diag
    F = np.zeros((nA, 3, nB, 3))
    F[ia, :, ib, :] = -blocks
    return F.reshape(3 * nA, 3 * nB), diag


def _add_diag_blocks(H, diag):
    n = len(diag)
    H.reshape(n, 3, n, 3)[np.arange(n), :, np.arange(n), :] += diag


def ring_labels(prot_xyz, memb_xyz, r0=20., dr=20.):
    """Labels the protein as layer 0 and splits the membrane into concentric rings 1, 2, ... around it.

    The first ring extends r0 beyond the farthest protein mass (in x-y) and the next ones are dr wide. A last ring
    with fewer than 20% of the nodes of the previous one is merged into it, and so is a first ring compared to the
    second one.

    Args:
        prot_xyz (numpy.ndarray): Coordinates of the protein masses (n x 3).
        memb_xyz (numpy.ndarray): Coordinates of the membrane nodes (m x 3).
        r0 (float, optional): Width of the first ring (Å).
        dr (float, optional): Width of the outer rings (Å).

    Returns:
        numpy.ndarray: Labels of the membrane nodes (m integers >= 1).
    """
    c0 = _xy_center(prot_xyz)
    sys_r = np.max(np.linalg.norm((prot_xyz - c0)[:, :2], axis=1))
    env_r = np.linalg.norm((memb_xyz - c0)[:, :2], axis=1)
    labels = np.clip(np.floor((env_r - (sys_r + r0)) / dr) + 1, 0, None).astype(int) + 1
    uniq = np.unique(labels)
    if len(uniq) >= 2:
        if np.sum(labels == uniq[-1]) < 0.2 * np.sum(labels == uniq[-2]):
            labels[labels == uniq[-1]] = uniq[-2]
    uniq = np.unique(labels)
    if len(uniq) >= 2:
        if np.sum(labels == uniq[0]) < 0.2 * np.sum(labels == uniq[1]):
            labels[labels == uniq[0]] = uniq[1]
    return np.unique(labels, return_inverse=True)[1].reshape(-1) + 1


def reduce_membrane_hessian(h_prot, prot_xyz, memb_xyz, cutoff=15., k=1., r0=20., dr=20., overwrite=False):
    """Integrates the membrane out of the protein-membrane Hessian with the recursive formula of Zhang et al. 2021.

    With the protein as layer 0 and the membrane rings as layers 1 to l, the Hessian is block tridiagonal with
    diagonal blocks H_i and couplings F_i between layers i and i+1. The reduced Hessian of layer i is
    H~_i = H_i - F_i H~_(i+1)^-1 F_i^T, starting from H~_l = H_l, and the reduced protein Hessian is H~_0.

    Args:
        h_prot (numpy.ndarray): The protein Hessian (3n x 3n).
        prot_xyz (numpy.ndarray): Coordinates of the protein masses (n x 3).
        memb_xyz (numpy.ndarray): Coordinates of the membrane nodes (m x 3).
        cutoff (float, optional): Cutoff of the membrane springs (Å).
        k (float, optional): Spring constant of the membrane springs.
        r0 (float, optional): Width of the first ring (Å).
        dr (float, optional): Width of the outer rings (Å).
        overwrite (bool, optional): If True, h_prot is modified in place to save memory.

    Returns:
        numpy.ndarray: The reduced protein Hessian (3n x 3n).
    """
    labels = ring_labels(prot_xyz, memb_xyz, r0, dr)
    layers = [prot_xyz] + [memb_xyz[labels == i] for i in range(1, labels.max() + 1)]
    reduced = None
    for i in reversed(range(len(layers))):
        X = layers[i]
        if i == 0:
            H = h_prot if overwrite else h_prot.copy()
        else:
            H = _intra_hessian(X, cutoff, k)
            _add_diag_blocks(H, _coupling(X, layers[i - 1], cutoff, k, with_F=False)[1])
        if reduced is not None:
            F, diag = _coupling(X, layers[i + 1], cutoff, k)
            _add_diag_blocks(H, diag)
            # F S^-1 F^T computed as Y^T Y with S = L L^T and Y = L^-1 F^T, which keeps H symmetric without a copy
            Y = np.linalg.solve(np.linalg.cholesky(reduced), F.T)
            del F
            H -= np.dot(Y.T, Y)
            del Y
        reduced = H
    return reduced
