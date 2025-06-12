import os
import re
import pickle

def load_data(which_slicer):
    if not which_slicer in ["our","ww","pump"]:
        msg = f"Supports only \"our\" (this slicer), \"ww\" (wvan-Woerdens), \"pump\" (summver slicer)"
        raise NotImplementedError(msg)
    
    path = f"./cvp_comp/{which_slicer}/"
    pattern = re.compile(
        r"""^cvp_comp_
            (?P<n>\d+)_
            (?P<lat_num>\d+)_
            (?P<inst_per_lat>\d+)_
            (?P<betamax>[^_]+)_
            (?P<appr_fact>\d+\.\d{4})  # matches floats like 0.1234
            \.pkl$
        """,
        re.VERBOSE
    )

    D = {}

    for fname in os.listdir(path):
        match = pattern.match(fname)
        if not match:
            continue

        gd = match.groupdict()
        n             = int(gd['n'])
        lat_num       = int(gd['lat_num'])
        inst_per_lat  = int(gd['inst_per_lat'])
        # if betamax is an integer, you can cast to int instead
        try:
            betamax = float(gd['betamax'])
        except ValueError:
            betamax = gd['betamax']    # leave as string if non-numeric
        appr_fact     = float(gd['appr_fact'])

        # load the pickle
        with open(os.path.join(path, fname), "rb") as f:
            content = pickle.load(f)

        D[n] = {
            "lat_num":      lat_num,
            "inst_per_lat": inst_per_lat,
            "betamax":      betamax,
            "appr_fact":    appr_fact,
            "content":      content
        }

        aggregated = {}

    for n, info in D.items():
        records = info.get('content', [])
        if not records:
            # skip empty
            continue

        sum_tslice = 0.0
        sum_tpump_plus_tslice = 0.0
        sum_dt_over_gh = 0.0

        for rec in records:
            # rec structure: [nrand, Tpump, Tslice, db_size, dt, gh]
            Tpump  = rec[1]
            Tslice = rec[2]
            dt      = rec[4]
            gh      = rec[5]

            sum_tslice += Tslice
            sum_tpump_plus_tslice += (Tpump + Tslice)
            sum_dt_over_gh += (dt / gh)

        count = len(records)
        aggregated[n] = {
            'avg_Tslice':               sum_tslice / count,
            'avg_Tpump_plus_Tslice':    sum_tpump_plus_tslice / count,
            'avg_dt_over_gh':           sum_dt_over_gh / count,
        }

    return aggregated

if __name__ == "__main__":
    D_our  = load_data("our")
    D_ww   = load_data("ww")
    D_pump = load_data("pump")

    print("    |           our           |          DLv20          |         summver         |")
    print("n   |      t     |     approx |      t     |     approx |      t     |     approx |")

    #  Define a format with:
    #    - n:    3-wide integer
    #    - t:    10-wide float, 4 decimals
    #    - approx: 10-wide float, 4 decimals
    fmt = "{:3d} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} |"

    # print header
    # print(fmt.format(
    #     0,   # dummy n for header
    #     0.0, # dummy t1
    #     0.0, # dummy approx1
    #     0.0, # dummy t2
    #     0.0, # dummy approx2
    #     0.0, # dummy t3
    #     0.0  # dummy approx3
    # ).replace("0.0000", "t").replace("0.0000", "approx", 1))

    for n in sorted( D_our.keys() ):
        try:
            t1, approx1, t2, approx2, t3, approx3  = D_our[n]["avg_Tslice"], D_our[n]["avg_dt_over_gh"],D_ww[n]["avg_Tslice"], D_ww[n]["avg_dt_over_gh"],D_pump[n]["avg_Tslice"], D_pump[n]["avg_dt_over_gh"],
            print(fmt.format(n, t1, approx1, t2, approx2, t3, approx3))
        except KeyError: #incomplete data
            pass