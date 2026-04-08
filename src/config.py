"""
Configuration for Gospel Steering Vectors.

Adapted from emotion-vectors for TinyBox Green (6x RTX 4090 24GB).
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ACTIVATION_DIR = PROJECT_DIR / "activations"
VECTOR_DIR = PROJECT_DIR / "vectors"

# Gospel text — either local data/ dir or the fine-tuning project
GOSPEL_DATA_DIR = PROJECT_DIR / "data" / "gospels"

# ── Model ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3.5-9B"
HIDDEN_DIM = 4096      # Qwen 3.5 9B hidden dimension
NUM_LAYERS = 48         # Qwen 3.5 9B has 48 decoder layers

# ── Hardware (DGX Spark: 1x GB10 Blackwell, 128GB unified) ──────────────
# 9B model in BF16 = ~18GB, fits easily in 128GB with 110GB headroom.
USE_QUANTIZATION = False  # Full BF16 on Sparks; set True for 4090s

# ── Extraction ───────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS = 256  # Target tokens per text chunk
MEAN_POOL_SKIP_TOKENS = 4  # Skip BOS + special tokens when mean-pooling

# ── Gospels ──────────────────────────────────────────────────────────────
GOSPELS = ["matthew", "mark", "luke", "john"]

GOSPEL_DESCRIPTIONS = {
    "matthew": "Jewish Messiah, fulfillment of prophecy, kingdom of heaven, "
               "the Law, righteousness, teaching authority, Sermon on the Mount",
    "mark":    "Suffering servant, urgency ('immediately'), messianic secret, "
               "action over discourse, cost of discipleship, power and weakness",
    "luke":    "Universal salvation, compassion for outcasts/women/poor, "
               "parables of mercy, joy, Holy Spirit, prayer, Magnificat",
    "john":    "Divine Word (Logos), high Christology, 'I am' statements, "
               "love, light vs darkness, eternal life, intimate union with God",
}

# ── PCA Denoising ────────────────────────────────────────────────────────
PCA_VARIANCE_THRESHOLD = 0.50  # Remove components explaining up to 50% of neutral variance

# ── Neutral text for PCA denoising ───────────────────────────────────────
# Emotionally and theologically neutral passages for computing background
# activation patterns to subtract from Gospel directions.
NEUTRAL_TEXTS = [
    "Water is composed of two hydrogen atoms and one oxygen atom, forming a molecule with the chemical formula H2O. It exists in three states of matter.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
    "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second, which is considered a fundamental constant of nature.",
    "Iron is a chemical element with symbol Fe and atomic number 26. It is the most common element by mass forming the outer and inner core of Earth.",
    "The process of cellular respiration converts glucose and oxygen into carbon dioxide, water, and adenosine triphosphate for cellular energy.",
    "Tectonic plates are massive segments of Earth's lithosphere that move, float, and sometimes fracture, and whose interaction causes continental drift.",
    "Binary code represents text or computer processor instructions using the binary number system's two symbols, typically zero and one.",
    "Gravity is a fundamental force of nature that attracts any two objects with mass. The force is proportional to the product of their masses.",
    "The periodic table organizes chemical elements by increasing atomic number, electron configuration, and recurring chemical properties.",
    "An algorithm is a finite sequence of well-defined instructions, typically used to solve a class of specific problems or to perform a computation.",
    "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells that generate most of the cell's supply of ATP.",
    "A semiconductor is a material that has electrical conductivity between that of a conductor and an insulator, used extensively in electronics.",
    "The Krebs cycle is a series of chemical reactions used by all aerobic organisms to release stored energy through the oxidation of acetyl-CoA.",
    "Ohm's law states that the current through a conductor between two points is directly proportional to the voltage across the two points.",
    "The electromagnetic spectrum is the range of frequencies of electromagnetic radiation and their respective wavelengths and photon energies.",
    "Boyle's law states that the pressure of a given mass of an ideal gas is inversely proportional to its volume at a constant temperature.",
    "A prime number is a natural number greater than one that is not a product of two smaller natural numbers other than one and itself.",
    "Newton's third law states that for every action there is an equal and opposite reaction, describing the forces between interacting bodies.",
    "The human genome contains approximately three billion base pairs of DNA organized into twenty-three pairs of chromosomes in the cell nucleus.",
    "Convection is the transfer of heat through the movement of fluids, where warmer portions rise and cooler portions sink due to density differences.",
    "The Fibonacci sequence begins with zero and one, and each subsequent number is the sum of the two preceding numbers in the sequence.",
    "Entropy is a measure of the number of possible arrangements the atoms in a system can have, often associated with disorder or randomness.",
    "A transistor is a semiconductor device used to amplify or switch electronic signals and electrical power, forming the basis of modern electronics.",
]
