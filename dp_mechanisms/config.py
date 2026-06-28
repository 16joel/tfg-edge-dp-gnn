"""
Configuració compartida dels mecanismes Edge-DP.

FONT ÚNICA DE VERITAT per al repartiment del pressupost de privadesa.

Tots els components del projecte (mecanismes, orquestradors, persistència de
resultats) han de llegir el repartiment d'aquí. Mai s'ha de tornar a
hardcodejar el split en cap altre fitxer: històricament això va provocar que
el codi (1/99), els prints (20/80), els docstrings (50/50) i els resums
guardats (50/50) declaressin repartiments diferents.

Justificació del valor actual (1% / 99%):
    S'han explorat empíricament diversos repartiments (50/50, 20/80, 1/99).
    La consulta de densitat té sensibilitat 1/C(n,2), extremadament petita,
    per la qual cosa una fracció mínima del pressupost (1%) ja produeix una
    estimació de densitat d_tilde prou precisa per a la majoria de règims;
    dedicar el 99% restant a la pertorbació estructural maximitza la
    informació topològica que sobreviu per a un mateix epsilon total.
"""

# Fracció d'epsilon dedicada a la pertorbació de la densitat global (epsilon1)
EPSILON_DENSITY_FRACTION = 0.01

# Fracció d'epsilon dedicada a la pertorbació estructural: RR o soroll de
# Laplace sobre l'adjacència (epsilon2)
EPSILON_STRUCTURE_FRACTION = 1.0 - EPSILON_DENSITY_FRACTION


def split_epsilon(epsilon: float):
    """
    Reparteix el pressupost total entre les dues consultes del mecanisme.

    Per composició seqüencial, el mecanisme complet és
    (epsilon1 + epsilon2) = epsilon - edge-DP.

    Parameters
    ----------
    epsilon : float
        Pressupost total de privadesa.

    Returns
    -------
    epsilon1 : float
        Pressupost per a la pertorbació de la densitat.
    epsilon2 : float
        Pressupost per a la pertorbació estructural (RR / Laplace).
    """
    epsilon1 = epsilon * EPSILON_DENSITY_FRACTION
    epsilon2 = epsilon * EPSILON_STRUCTURE_FRACTION
    return epsilon1, epsilon2


def budget_split_description() -> str:
    """
    Descripció textual del repartiment, per a logs, summary.txt i metadata.txt.
    Es genera a partir de les constants per garantir que el que es registra
    coincideix sempre amb el que s'executa.
    """
    return (
        f"{EPSILON_DENSITY_FRACTION:.0%} of epsilon for density perturbation "
        f"(epsilon1), {EPSILON_STRUCTURE_FRACTION:.0%} for structural "
        f"perturbation (epsilon2)"
    )
