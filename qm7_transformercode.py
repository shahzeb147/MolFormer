# Find Broken Views
import numpy as np
np.random.seed(42)

def small_views(vs, piece_size=4):
    """
    Split views into smaller pieces such that each small view contains only atomic number + 3D coordinates.

    Parameters:
    - vs: numpy array of views
    - piece_size: size of each small view (default = 4: atomic number + 3D coordinates)
    
    Returns:
    - new_view: numpy array with views split into atomic pieces
    """
    # Check that the view size is divisible by the piece size
    if vs.shape[-1] % piece_size != 0:
        raise ValueError(f"View size {vs.shape[-1]} is not divisible by piece size {piece_size}.")
    
    single_atom_piece = vs.shape[-1] // piece_size
    
    new_view = np.reshape(vs, (vs.shape[0], vs.shape[1], single_atom_piece, piece_size))
    
    return new_view

###################################################################################

# Switch for atomic properties

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

#######################################################################################


# calcualte coulomb matrix

def coulomb_interaction(nuclear_charge_i, nuclear_charge_j, coord_i, coord_j):
    """
calculate coulomb interaction
Parameters:
- nuclear_charge_i: nuclear charge of atom i
- nuclear_charge_j: nuclear charge of atom j
-coord_i: 3D coordinates of atom i
-coord_j: 3D coordinates of atom j

Returns: 
- Coulomb interaction 

    """

    # calculate distance between two atoms

    distance = np.linalg.norm(np.array(coord_i)-np.array(coord_j))

    if distance == 0:
        return 0
    return(nuclear_charge_i * nuclear_charge_j)/ distance

##########################################################################################

def coulomb_interaction_broken(single_atom):
    """
    calculate coulomb interaction of an atom with all other atoms present in a view
    """
    number_of_molecules = single_atom.shape[0]
    number_of_views = single_atom.shape[1]
    number_of_atoms = single_atom.shape[2]

    interaction = np.zeros((number_of_molecules, number_of_views, number_of_atoms, number_of_atoms-1))

# extract a single view data first

    for mol_index in range(number_of_molecules):
        for view_index in range(number_of_views):
            view = single_atom[mol_index, view_index]

# extract specific atom and then find its coulomb interaction
            for i in range(number_of_atoms):
                nuclear_charge_i = view[i, 0]
                coord_i = view[i, 1:4]

                col_index = 0
                for j in range(number_of_atoms):
                    if i != j:
                        nuclear_charge_j = view[j, 0]
                        coord_j = view[j, 1:4]
                        interaction[mol_index, view_index, i, col_index] = coulomb_interaction(nuclear_charge_i, nuclear_charge_j, coord_i, coord_j)
                        col_index += 1
    return interaction

#################################################################################################

atomic_number_to_type = {
    1.0: "H", 
    6.0: "C", 
    7.0: "N", 
    8.0: "O", 
    9.0: "F",
    16.0: "S"
}
def get_embeddings(single_atom, coulomb_interaction_all, atom_properties, single_atomic_property_switches):

    """
    get features for each atom

    parameters: 
    - single_atom: broken views of size (80,17,,17,4) and (20,17,,17,4)
    - coulomb_interaction_all: coulomb interaction of size (80, 17,17,16) and (20, 17,17,16)
    - atom_properties: dictionary containing properties for each atom type
    - single_atomic_properties_switches: switches to add single atom properties

    Returns: 
    - array with atomic number, 3D coordinates, atomic properties and coulomb interactions
    """
    



    number_of_molecules = single_atom.shape[0]
    number_of_views = single_atom.shape[1]
    number_of_atoms = single_atom.shape[2]
    property_keys = [prop for prop, is_on in single_atomic_property_switches.items() if is_on]
    num_properties = len(property_keys)

    atom_embeddings = []

    for mol_id in range(number_of_molecules):
        molecule_extended = []
        for view_id in range(number_of_views):
            view1 = single_atom[mol_id, view_id]
            view_extended = []

            for atom_id in range(number_of_atoms):
                atomic_number = view1[atom_id, 0]

                # handle atoms with atomic number 0.0
                if atomic_number == 0.0:
                    extended_atom = np.zeros(1 + 3 + num_properties + 22)  # fill with zeros
                else:
                    coord = view1[atom_id, 1:4]
                    atom_type = atomic_number_to_type[atomic_number]
                    properties = [atom_properties[atom_type][prop] for prop in property_keys]
                    coulomb_interaction = coulomb_interaction_all[mol_id, view_id, atom_id]

                    extended_atom = np.concatenate([[atomic_number], coord, properties, coulomb_interaction]) #concatenate atomic number, coord, properties and coulomb matrix of a single atom

                view_extended.append(extended_atom) #conatin feature for all atoms in a single view (23 in our case)
            molecule_extended.append(np.array(view_extended)) #all views in a molecule

        atom_embeddings.append(np.array(molecule_extended))

    return np.array(atom_embeddings)


##############################################################################