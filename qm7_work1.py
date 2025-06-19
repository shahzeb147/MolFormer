# Switch for atomic properties
import numpy as np
np.random.seed(42)
# Dictionary containing properties for each atom type
atom_properties = {
    "H": {"electronegativity": 2.20, "atomic_mass": 1.008, "valence_electrons": 1},   # Hydrogen (atomic number 1)
    "C": {"electronegativity": 2.55, "atomic_mass": 12.01, "valence_electrons": 4},  # Carbon (atomic number 6)
    "N": {"electronegativity": 3.04, "atomic_mass": 14.007, "valence_electrons": 5}, # Nitrogen (atomic number 7)
    "O": {"electronegativity": 3.44, "atomic_mass": 15.999, "valence_electrons": 6}, # Oxygen (atomic number 8)
    "S": {"electronegativity": 2.58, "atomic_mass": 32.06, "valence_electrons": 6}   # Sulfur (atomic number 16)
}

# switch for atomic properties

# more properties can go here 
single_atomic_property_switches = {
    "electronegativity": True,
    "atomic_mass": True,
    "valence_electrons": True
}

#############################################

atomic_number_to_type = {
    1.0: "H", 
    6.0: "C", 
    7.0: "N", 
    8.0: "O", 
    9.0: "F",
    16.0: "S"
}



def get_embeddings(single_atom, atom_properties, single_atomic_property_switches, embedding_size=16):
    """
    Get fixed-size embeddings (length=16) for each atom using atomic number, 3D coordinates, and selected properties.
    Remaining positions are zero-padded.

    Parameters:
    - single_atom: numpy array (N_molecules, N_views, N_atoms, 4)
    - atom_properties: dictionary of properties per atom type
    - single_atomic_property_switches: which properties to include
    - embedding_size: desired fixed embedding length (default = 16)

    Returns:
    - atom_embeddings: numpy array (N_molecules, N_views, N_atoms, embedding_size)
    """
    number_of_molecules = single_atom.shape[0]
    number_of_views = single_atom.shape[1]
    number_of_atoms = single_atom.shape[2]
    
    property_keys = [prop for prop, is_on in single_atomic_property_switches.items() if is_on]
    num_features = 1 + 3 + len(property_keys)

    if num_features > embedding_size:
        raise ValueError(f"Embedding size {embedding_size} is too small for selected features (requires at least {num_features}).")

    atom_embeddings = []

    for mol_id in range(number_of_molecules):
        molecule_extended = []
        for view_id in range(number_of_views):
            view = single_atom[mol_id, view_id]
            view_extended = []

            for atom_id in range(number_of_atoms):
                atomic_number = view[atom_id, 0]

                if atomic_number == 0.0:
                    extended_atom = np.zeros(embedding_size)
                else:
                    coord = view[atom_id, 1:4]
                    atom_type = atomic_number_to_type.get(atomic_number, "H")  # default to H if unknown
                    properties = [atom_properties[atom_type][prop] for prop in property_keys]
                    raw_features = np.concatenate([[atomic_number], coord, properties])

                    # Pad with zeros to reach desired embedding size
                    padded = np.zeros(embedding_size)
                    padded[:len(raw_features)] = raw_features
                    extended_atom = padded

                view_extended.append(extended_atom)
            molecule_extended.append(np.array(view_extended))
        atom_embeddings.append(np.array(molecule_extended))

    return np.array(atom_embeddings)
