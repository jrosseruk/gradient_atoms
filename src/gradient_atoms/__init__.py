"""Gradient Atoms: Unsupervised discovery of model behaviors via sparse decomposition of training gradients."""

from gradient_atoms.extract import extract_gradients_single_gpu
from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac, unproject_atom
from gradient_atoms.dictionary import run_dictionary_learning, characterise_atoms, extract_keywords
from gradient_atoms.steering import create_steered_adapter, kill_gpu, start_vllm, eval_model
from gradient_atoms.plotting import ATOM_LABELS, load_atom_data, embed_2d, plot_atoms
