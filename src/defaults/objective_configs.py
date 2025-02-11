"""A collection of metric selections for specific objectives."""

from ..criteria.classifier_objectives import UncertaintyThreshold
from ..criteria.image_comparison import CFrobeniusDistance
from ..criteria.objective_functions import (
    DynamicConfidenceBalance,
    IsMisclassified,
    NaiveConfidenceBalance,
)

"""
### ADVERSARIAL TESTING:
When doing adversarial testing we aim to find inputs that are close to the original, but perturb the classifiers predictions.

In the untargeted case we do not care what class the perturbed input fall into.
In the targeted case we want to find a perturbed input with a specific (secondary) class.

Note that generally this can be a discrete problem where we check, if misclassified. 
But for optimization continuous problems produce better results, therefore we use confidence imbalance.
"""
UNTARGETED_ADVERSARIAL_TESTING = [
    CFrobeniusDistance(),
    DynamicConfidenceBalance(inverse=True, target_primary=False),
]
TARGETED_ADVERSARIAL_TESTING = [
    CFrobeniusDistance(),
    NaiveConfidenceBalance(inverse=True, target_primary=False),
]

"""
### Boundary TESTING:
Boundary testing essentially tries to find inputs that confuse the classifier, i.e. where the confidence of 2 or n classes are in a balance.

This again can be untargeted, where we do not care about the final class or targeted where we want to find a specific boundary.
Note that here we dont care about the distance of images, since the boundary can either be of an adversarial subset in the classifiers desicion-manifold, or it can be a different class set entirely.
"""
UNTARGETED_BOUNDARY_TESTING = [DynamicConfidenceBalance()]
TARGETED_BOUNDARY_TESTING = [NaiveConfidenceBalance()]
ADVERSARIAL_BOUNDARY_TESTING = [NaiveConfidenceBalance(), CFrobeniusDistance()]

"""
### DIVERSITY SAMPLING:
Diversity testing aims to find inputs that are dissimilar to the provided data, but are inferred by the classifer.

Here we want to find inputs that are classified correctly but have a high distance to the initial input.
"""
DIVERSITY_SAMPLING = [CFrobeniusDistance(inverse=True), IsMisclassified()]


"""
### Validity Boundary Testing.
Here we want to find the boundary to the validity domain in the classifier. As this is not formalizable we approximate it here.
"""
VALIDITY_BOUNDARY_TESTING = [UncertaintyThreshold(0.1, absolute=True), IsMisclassified()]
