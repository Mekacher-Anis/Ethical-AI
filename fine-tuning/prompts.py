BASE_PROMPT_TMPL = """
Based on the following definition decide if a given argument contains {attribute_name}. {attribute_name} is defined as: {attribute_defintion}.
With that definition decide if the following argument contains {attribute_name}:
The topic of the following discussion is: {topic}
---------------------
{argument}
---------------------
Does the argument contain {attribute_name}? True or False? {true_or_false}
"""
ATTRIBUTE_DEFINITIONS = {
    # "Toxic Emotions": "An argument has toxic emotions if the emotions appealed to are deceptive or their intensities do not provide room for critical evaluation of the issue by the reader",
    "Excessive Intensity": "The emotions appealed to by an argument are unnecessarily strong for the discussed issue",
    "Emotional Deception": "The emotions appealed to are used as deceptive tricks to win, derail, or end the discussion",
    # "Missing Commitment": "An argument is missing commitment if the issue is not taken seriously or openness other’s arguments is absent",
    "Missing Seriousness": "The argument is either trolling others by suggesting (explicitly or implicitly) that the issue is not worthy of being discussed or does not contribute meaningfully to the discussion",
    "Missing Openness": "The argument displays an unwillingness to consider arguments with opposing viewpoints and does not assess the arguments on their merits but simply rejects them out of hand",
    # "Missing Intelligibility": "An argument is not intelligible if its meaning is unclear or irrelevant to the issue or if its reasoning is not understandable",
    "Unclear Meaning": "The argument’s content is vague, ambiguous, or implicit, such that it remains unclear what is being said about the issue (it could also be an unrelated issue)",
    "Missing Relevance": "The argument does not discuss the issue, but derails the discussion implicitly towards a related issue or shifts completely towards a different issue",
    "Confusing Reasoning": "The argument’s components (claims and premises) seem not to be connected logically",
    # "Other Reasons": "An argument is inappropriate if it contains severe orthographic errors or for reasons not covered by any other dimension",
    "Detrimental Orthography": "The argument has serious spelling and/or grammatical errors, negatively affecting its readability",
    "Reason Unclassified": "There are any other reasons than those above for why the argument should be considered inappropriate",
}
