from pydantic import BaseModel, Field, constr
from typing import List, Dict, Optional

# --- SCPN Layer & Identity Schemas (v2.1 - Formalism Update) ---

class ClassicalIdentity(BaseModel):
    """ D-Script: Physical, classical description. """
    Structure: str
    Function: str
    Pathology: Optional[str] = None

class SCPN_Map(BaseModel):
    """
    I-Script: The 16-Layer Informational Mapping.
    This now correctly maps to Papers 1-16.
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
    """ S-Script: Metaphysical and vibrational identity. """
    SymbolicName: str
    VIBRANA_Key: Optional[str] = None
    MetaphysicalMapping: Optional[str] = None

class Metadata(BaseModel):
    """ M-Script: Versioning and status. """
    Version: str = "0.1.0"
    Status: str = "Draft"
    LogReference: Optional[str] = None # e.g., "Paper 21, p. 1303"

class ValidationPass(BaseModel):
    """
    V-Script: Falsification hypotheses and experimental/simulation references
    per the Part III (Papers 17-20) Validation Suite.
    """
    Falsification_Hypotheses: Optional[List[str]] = Field(None, description="Testable hypotheses to *disprove* the SCPN mappings.")
    Experimental_Protocols: Optional[List[str]] = Field(None, description="References to protocols in Paper 17.")
    Simulation_Suite: Optional[List[str]] = Field(None, description="References to simulation models in Paper 18.")

# --- NEWLY ADDED FORMALISM SCHEMA ---
class VIBRANA_Interface(BaseModel):
    Input_Verbs: Optional[List[str]] = None
    Output_Nouns: Optional[List[str]] = None

class FormalismPass(BaseModel):
    """
    F-Script: The mathematical, computational, and linguistic formalism
    of the monad (per Papers 19-20).
    """
    KeyEquations_LaTeX: Optional[List[str]] = Field(None, description="The core equations defining the monad's function.")
    SimulationModel_Ref: Optional[str] = Field(None, description="Path/Reference to the Python simulation code.")
    VIBRANA_Interface: Optional[VIBRANA_Interface] = Field(None, description="The VIBRANA language interface.")

# --- The Core Monad Class (UPDATED) ---

class AnatomicalMonad(BaseModel):
    """
    This is the core data structure for the Holonomic Atlas (Paper 21).
    It is now validated against the complete SCPN corpus (Papers 0-16).
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