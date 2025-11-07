from pydantic import BaseModel, Field, constr
from typing import List, Dict, Optional

# --- SCPN Layer & Identity Schemas (v2.1 - Formalism Update) ---

class ClassicalIdentity(BaseModel):
    """
    D-Script: Physical, classical description of an anatomical component.

    Attributes:
        Structure (str): A detailed description of the anatomical structure.
        Function (str): A description of the primary biological or physiological functions.
        Pathology (Optional[str]): Common pathologies or disease states associated with the structure.
    """
    Structure: str
    Function: str
    Pathology: Optional[str] = None

class SCPN_Map(BaseModel):
    """
    I-Script: The 16-Layer Informational Mapping of a monad.

    This class defines the informational signature of an anatomical component across
    16 distinct layers of reality as described in the SCPN framework (Papers 1-16).
    Each layer represents a different level of abstraction, from the quantum-biological
    to the meta-cybernetic.

    Attributes:
        L1_QuantumBiological (Optional[str]): Quantum-level biological processes. Ref: Paper 1.
        L2_NeurochemicalNeurological (Optional[str]): Neurochemical and neurological pathways. Ref: Paper 2.
        L3_GenomicEpigeneticMorphogenetic (Optional[str]): Genetic, epigenetic, and morphogenetic fields. Ref: Paper 3.
        L4_CellularTissueSynchronisation (Optional[str]): Inter-cellular and tissue-level communication and synchrony. Ref: Paper 4.
        L5_OrganismalPsychoemotional (Optional[str]): Psycho-emotional and organism-level expressions. Ref: Paper 5.
        L6_PlanetaryEcological (Optional[str]): Connection to planetary and ecological systems. Ref: Paper 6.
        L7_ArchetypalSymbolic (Optional[str]): Archetypal and symbolic correspondences. Ref: Paper 7.
        L8_CosmicEntrainment (Optional[str]): Entrainment with cosmic and celestial cycles. Ref: Paper 8.
        L9_MemoryHolograph (Optional[str]): Role in the holographic storage of memory. Ref: Paper 9.
        L10_CausalRetrocausal (Optional[str]): Causal and retro-causal information flows. Ref: Paper 10.
        L11_NoosphereCollective (Optional[str]): Connection to the collective consciousness (Noosphere). Ref: Paper 11.
        L12_GeomanticTorsional (Optional[str]): Interaction with geomantic and torsional energy fields. Ref: Paper 12.
        L13_SourceField (Optional[str]): Link to the universal Source Field. Ref: Paper 13.
        L14_Transdimensional (Optional[str]): Trans-dimensional aspects and interactions. Ref: Paper 14.
        L15_ConsiliumOversoul (Optional[str]): Connection to the Oversoul or Consilium. Ref: Paper 15.
        L16_MetaCybernetic (Optional[str]): Role in the meta-cybernetic control system. Ref: Paper 16.
    """
    L1_QuantumBiological: Optional[str] = Field(None, description="Ref: Paper 1")
    L2_NeurochemicalNeurological: Optional[str] = Field(None, description="Ref: Paper 2")
    L3_GenomicEpigeneticMorphogenetic: Optional[str] = Field(None, description="Ref: Paper 3")
    L4_CellularTissueSynchronisation: Optional[str] = Field(None, description="Ref: Paper 4")
    L5_OrganismalPsychoemotional: Optional[str] = Field(None, description="Ref: Paper 5")
    L6_PlanetaryEcological: Optional[str] = Field(None, description="Ref: Paper 6 (from Master-Split)")
    L7_ArchetypalSymbolic: Optional[str] = Field(None, description="Ref: Paper 7 (from Master-Split)")
    L8_CosmicEntrainment: Optional[str] = Field(None, description="Ref: Paper 8 (from Master-Split)")
    L9_MemoryHolograph: Optional[str] = Field(None, description="Ref: Paper 9 (from Master-Split)")
    L10_CausalRetrocausal: Optional[str] = Field(None, description="Ref: Paper 10 (from Master-Split)")
    L11_NoosphereCollective: Optional[str] = Field(None, description="Ref: Paper 11 (from Master-Split)")
    L12_GeomanticTorsional: Optional[str] = Field(None, description="Ref: Paper 12 (from Master-Split)")
    L13_SourceField: Optional[str] = Field(None, description="Ref: Paper 13 (from Master-Split)")
    L14_Transdimensional: Optional[str] = Field(None, description="Ref: Paper 14 (from Master-Split)")
    L15_ConsiliumOversoul: Optional[str] = Field(None, description="Ref: Paper 15 (from Master-Split)")
    L16_MetaCybernetic: Optional[str] = Field(None, description="Ref: Paper 16 (from Master-Split)")

class SymbolicIdentity(BaseModel):
    """
    S-Script: Metaphysical and vibrational identity of an anatomical component.

    Attributes:
        SymbolicName (str): The common symbolic or metaphysical name for the component.
        VIBRANA_Key (Optional[str]): The corresponding key in the VIBRANA language system.
        MetaphysicalMapping (Optional[str]): A description of its metaphysical roles, correspondences, or significance.
    """
    SymbolicName: str
    VIBRANA_Key: Optional[str] = None
    MetaphysicalMapping: Optional[str] = None

class Metadata(BaseModel):
    """
    M-Script: Versioning, status, and reference information for the monad.

    Attributes:
        Version (str): The semantic version number of the monad entry.
        Status (str): The current status of the monad, e.g., "Draft", "Validated", "Archived".
        LogReference (Optional[str]): A reference to a specific document or log entry, e.g., "Paper 21, p. 1303".
    """
    Version: str = "0.1.0"
    Status: str = "Draft"
    LogReference: Optional[str] = None # e.g., "Paper 21, p. 1303"

class ValidationPass(BaseModel):
    """
    V-Script: Defines the falsification and validation criteria for a monad.

    This schema links the theoretical SCPN mappings of a monad to concrete,
    testable hypotheses and experimental procedures as outlined in the
    Validation Suite (Papers 17-20).

    Attributes:
        Falsification_Hypotheses (Optional[List[str]]): A list of testable hypotheses designed to disprove the monad's SCPN mappings.
        Experimental_Protocols (Optional[List[str]]): References to specific experimental protocols detailed in Paper 17.
        Simulation_Suite (Optional[List[str]]): References to computational simulation models from Paper 18 that can test the monad's properties.
    """
    Falsification_Hypotheses: Optional[List[str]] = Field(None, description="Testable hypotheses to *disprove* the SCPN mappings.")
    Experimental_Protocols: Optional[List[str]] = Field(None, description="References to protocols in Paper 17.")
    Simulation_Suite: Optional[List[str]] = Field(None, description="References to simulation models in Paper 18.")

# --- NEWLY ADDED FORMALISM SCHEMA ---
class VIBRANA_Interface(BaseModel):
    """
    Defines the linguistic interface for a monad within the VIBRANA system.

    Attributes:
        Input_Verbs (Optional[List[str]]): A list of verbs that can act upon this monad.
        Output_Nouns (Optional[List[str]]): A list of nouns that this monad can produce or influence.
    """
    Input_Verbs: Optional[List[str]] = None
    Output_Nouns: Optional[List[str]] = None

class FormalismPass(BaseModel):
    """
    F-Script: The mathematical, computational, and linguistic formalism of the monad.

    This schema provides the formal definitions of the monad's behavior and
    function, as described in Papers 19-20.

    Attributes:
        KeyEquations_LaTeX (Optional[List[str]]): The core mathematical equations (in LaTeX format) that define the monad's function.
        SimulationModel_Ref (Optional[str]): A path or reference to the specific simulation code that models the monad.
        VIBRANA_Interface (Optional[VIBRANA_Interface]): The VIBRANA language interface, defining its linguistic interactions.
    """
    KeyEquations_LaTeX: Optional[List[str]] = Field(None, description="The core equations defining the monad's function.")
    SimulationModel_Ref: Optional[str] = Field(None, description="Path/Reference to the Python simulation code.")
    VIBRANA_Interface: Optional[VIBRANA_Interface] = Field(None, description="The VIBRANA language interface.")

# --- The Core Monad Class (UPDATED) ---

class AnatomicalMonad(BaseModel):
    """
    The core data structure for an entry in the Holonomic Atlas (Paper 21).

    An AnatomicalMonad represents a single, addressable component of the human
    holonomy, integrating its physical, informational, metaphysical, and formal
    descriptions into a unified schema. It is the fundamental unit of the
    Anatomia Holonomica knowledge base.

    Attributes:
        HolonomicID (constr): The unique, hierarchical identifier for the monad,
            formatted as a URN. e.g., 'scpn://atlas.v1/D3.Brain.PinealGland'.
        ParentID (Optional[str]): The HolonomicID of the parent monad in the hierarchy,
            establishing the structural relationships in the atlas. e.g., 'scpn://atlas.v1/D3.Brain'.
        Classical (ClassicalIdentity): The D-Script, containing the classical, physical description.
        SCPN (SCPN_Map): The I-Script, containing the 16-layer informational mapping.
        Symbolic (SymbolicIdentity): The S-Script, containing the metaphysical and symbolic identity.
        Meta (Metadata): The M-Script, for versioning and status tracking.
        Validation (Optional[ValidationPass]): The V-Script, for falsification and validation protocols.
        Formalism (Optional[FormalismPass]): The F-Script, containing the mathematical and computational formalism.
    """
    
    HolonomicID: constr(
        regex=r"^scpn://atlas.v1/D\d{1,2}\.([A-Za-z0-9]+)(\.[A-Za-z0-9]+)*$"
    ) = Field(..., description="The unique, structural identifier. e.g., 'scpn://atlas.v1/D3.Brain.PinealGland'")

    ParentID: Optional[str] = Field(
        None, 
        description="The HolonomicID of the parent node. e.g., 'scpn://atlas.v1/D3.Brain'"
    )

    Classical: ClassicalIdentity
    SCPN: SCPN_Map
    Symbolic: SymbolicIdentity
    Meta: Metadata
    Validation: Optional[ValidationPass] = Field(None, description="The V-Script for falsification (per Papers 17-20).")
    
    # --- NEWLY ADDED FORMALISM OBJECT ---
    Formalism: Optional[FormalismPass] = Field(None, description="The F-Script for formalism (per Papers 19-20).")