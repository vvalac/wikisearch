import random

FACTS = [
    "Crabs have two claws called chelae — the larger is the crusher and the smaller is the cutter.",
    "Fiddler crabs have one claw up to twice the size of their body, used only for signaling.",
    "The coconut crab is the largest land-dwelling arthropod, with a leg span up to one meter.",
    "Crabs taste and smell through tiny hairs on their legs and claws called setae.",
    "Some crabs carry sea anemones on their claws as a living chemical weapon.",
    "The horseshoe crab is not a true crab — it's more closely related to spiders and scorpions.",
    "Pea crabs are the world's smallest crabs and live inside oysters and mussels.",
    "Crabs are decapods — they have ten limbs, two of which evolved into claws.",
    "The Japanese spider crab has the longest leg span of any arthropod: up to 3.7 meters.",
    "Ghost crabs can run at nearly 1.6 meters per second and change direction instantly.",
    "Christmas Island red crabs migrate in columns 100 million strong to the sea each year.",
    "Crabs communicate by drumming or waving their claws — each species has a distinct pattern.",
    "The pistol shrimp snapping claw produces a cavitation bubble hotter than the surface of the sun. Crabs respect this.",
    "Mantis shrimp can punch hard enough to shatter glass — crabs maintain a healthy distance.",
    "Blue crabs can regenerate lost claws, though the new one is typically smaller.",
]


def random_fact() -> str:
    return random.choice(FACTS)
