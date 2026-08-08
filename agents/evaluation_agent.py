from difflib import SequenceMatcher

def evaluate(old_output, new_output):

    similarity = SequenceMatcher(
        None,
        old_output,
        new_output
    ).ratio()

    return round(similarity,2)