with open("dataset/company_rule.txt", "r", encoding="utf-8") as f:

    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) >0:
            print(line)