from pipeline import run_regression

with open("prompts/prompt_v1.txt") as f:
    p1 = f.read()

with open("prompts/prompt_v2.txt") as f:
    p2 = f.read()

results = run_regression(p1, p2)

print(results)