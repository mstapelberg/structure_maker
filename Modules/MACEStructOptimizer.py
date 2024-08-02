from __future__ import annotations

import contextlib
import inspect
import io
import pickle
import sys
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes, all_properties
from typing import Literal
from mace.calculators import MACECalculator
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.filters import FrechetCellFilter, Filter

if TYPE_CHECKING:
    from ase.io import Trajectory
    from ase.optimize.optimize import Optimizer


OPTIMIZERS = {
    "FIRE": FIRE,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
}

class MACEStructOptimizer:
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        #model: CHGNet | CHGNetCalculator | None = None,
        model: MACECalculator | None = None,
        optimizer_class: Optimizer | str | None = "LBFGS",
        use_device: str | None = None,
        stress_weight: float = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    ) -> None:
        """Provide a trained CHGNet model and an optimizer to relax crystal structures.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
        """
        
        
        if isinstance(optimizer_class, str):
            if optimizer_class in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[optimizer_class]
            else:
                raise ValueError(
                    f"Optimizer instance not found. Select from {list(OPTIMIZERS)}"
                )

        self.optimizer_class: Optimizer = optimizer_class
        

        if isinstance(model, MACECalculator):
            self.calculator = model
        else:
            print("model is not MACECalculator")


    def relax(
        self,
        atoms: Structure | Atoms,
        fmax: float | None = 0.01,
        steps: int | None = 500,
        relax_cell: bool | None = True,
        ase_filter: str | Filter = FrechetCellFilter,
        save_path: str | None = None,
        loginterval: int | None = 1,
        crystal_feas_save_path: str | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            ase_filter (str | ase.filters.Filter): The filter to apply to the atoms
                object for relaxation. Default = FrechetCellFilter
                Used to default to ExpCellFilter but was removed due to bug reported in
                https://gitlab.com/ase/ase/-/issues/1321 and fixed in
                https://gitlab.com/ase/ase/-/merge_requests/3024.
            save_path (str | None): The path to save the trajectory.
                Default = None
            loginterval (int | None): Interval for logging trajectory and crystal feas
                Default = 1
            crystal_feas_save_path (str | None): Path to save crystal feature vectors
                which are logged at a loginterval rage
                Default = None
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        if isinstance(ase_filter, str):
            try:
                import ase.filters

                ase_filter = getattr(ase.filters, ase_filter)
            except AttributeError as exc:
                valid_filter_names = [
                    name
                    for name, cls in inspect.getmembers(ase.filters, inspect.isclass)
                    if issubclass(cls, Filter)
                ]
                raise ValueError(
                    f"Invalid {ase_filter=}, must be one of {valid_filter_names}. "
                ) from exc
        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
            # atoms = atoms.to_ase_atoms()

        atoms.calc = self.calculator  # assign model used to predict forces

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)

            if crystal_feas_save_path:
                cry_obs = CrystalFeasObserver(atoms)

            if relax_cell:
                atoms = ase_filter(atoms)
            optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=loginterval)

            if crystal_feas_save_path:
                optimizer.attach(cry_obs, interval=loginterval)

            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if crystal_feas_save_path:
            cry_obs.save(crystal_feas_save_path)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)
        for key in struct.site_properties:
            struct.remove_site_property(property_name=key)
        #struct.add_site_property(
            #"magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
        #)
        return {"final_structure": struct, "trajectory": obs}
    


class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        #self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        #self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            #"magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class CrystalFeasObserver:
    """CrystalFeasObserver is a hook in the relaxation and MD process that saves the
    intermediate crystal feature structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a CrystalFeasObserver from an Atoms object."""
        self.atoms = atoms
        self.crystal_feature_vectors: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record Atoms crystal feature vectors after an MD/relaxation step."""
        self.crystal_feature_vectors.append(self.atoms._calc.results["crystal_fea"])

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)

    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to file."""
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)

