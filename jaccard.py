def jaccard_similarity(claim, evidence):
    claim_set = set(claim.split())
    evidence_set = set(evidence.split())
    print(claim_set)
    print(evidence_set)
    overlap = 1.0*len(claim_set.intersection(evidence_set))/len(claim_set)
    return overlap
claim = 'The Ten Commandments is an epic film.'
evidence = 'The film was listed as the tenth best film in the epic genre .'
jac = jaccard_similarity(claim, evidence)
print(jac)
