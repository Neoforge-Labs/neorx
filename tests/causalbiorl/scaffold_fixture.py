"""The twelve hardcoded strings the deleted _fallback_generate returned.

Pinned so tests can assert the environment never emits them again. Do not
import this anywhere outside tests.
"""

DELETED_FALLBACK_SCAFFOLDS = frozenset({
    "c1ccc2[nH]c(-c3ccncc3)nc2c1",
    "O=C(NCc1ccccc1)c1cc2ccccc2[nH]1",
    "Cc1nc2ccccc2n1Cc1ccc(F)cc1",
    "O=C(c1ccc(O)cc1)c1ccc(O)cc1O",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccc(-c2nc3ccccc3s2)cc1",
    "O=c1[nH]c2ccccc2c2ccccc12",
    "NC(=O)c1cccc(-c2cccnc2)c1",
    "Oc1ccc(-c2cc(-c3ccc(O)cc3)no2)cc1",
    "CC1=NN(c2ccccc2)C(=O)C1",
    "c1ccc(CNc2ncnc3[nH]cnc23)cc1",
    "CC(C)c1nnc(C(C)C)n1C1CC1c1ccc(F)cc1",
})
