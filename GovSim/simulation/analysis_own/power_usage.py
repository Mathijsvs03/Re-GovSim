
def get_stats(path="power_usage_stats"):
    stats = {}
    cols = []
    with open(path) as file:
        cols = list(map(list, zip(*[line.rstrip().split(";") for line in file])))
    for col in cols:
        stats[col[0]] = col[1:]
    return stats

if __name__ == "__main__":
    stats = get_stats()

    totals = {"Time": 0.0, "Energy": 0.0}
    for i in range(len(stats["JOB-STEP-AID"])):
        if not "sb" in stats["JOB-STEP-AID"][i]:
            continue
        totals["Time"] += float(stats["TIME(s)"][i])
        totals["Energy"] += float(stats["ENERGY(J)"][i])

    print(totals)