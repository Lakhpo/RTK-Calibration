import pyswmm
from pyswmm import Simulation, Nodes, SimulationPreConfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from datetime import datetime, timedelta
import math
import deap
from deap import base, creator, tools, algorithms
import sklearn
from sklearn import metrics


# From_excel_ordinal is used to get the serial date numbers from the input csv file from SSOAP
# and bring it into python
def from_excel_ordinal(ordinal: float, _epoch0=datetime(1899, 12, 31)) -> datetime:
    if ordinal >= 60:
        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!
    return (_epoch0 + timedelta(days=ordinal)).replace(microsecond=0)


def RSE(y_true, y_predicted):
    """
    - y_true: Actual values
    - y_predicted: Predicted values
    """
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))

    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse


def ConvertToLotus(times, _epoch0=datetime(1899, 12, 31)):
    LotusTimes = np.array([])
    for i in times:
        timediff = i - _epoch0
        Lotus = timediff.total_seconds() / 86400
        LotusTimes = np.append(LotusTimes, Lotus)
    return LotusTimes


def update_rtk_columns(
    new_rtk_list,
    inp_file_path="C:/Users/chase/Downloads/TEST.inp",
    hydrograph_name="Analysis_2008-03-03",
):
    """
    Updates RTK values (columns 47, 56, 65) for a given hydrograph,
    while preserving all other formatting/content.

    Arguments:
        inp_file_path: path to original .inp file
        hydrograph_name: name of the hydrograph (e.g., 'RDII_1')
        new_rtk_list: list of (R, T, K) floats; up to 3 triplets
    """
    with open(inp_file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    in_hydrograph_section = False
    replacing = False
    replace_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped.upper() == "[HYDROGRAPHS]":
            in_hydrograph_section = True
            updated_lines.append(line)
            continue

        if in_hydrograph_section:
            if stripped == "" or stripped.startswith("["):
                # End of section
                in_hydrograph_section = False
                replacing = False
                replace_count = 0
                updated_lines.append(line)
                continue

            if stripped.startswith(hydrograph_name):
                if not replacing:
                    # First line (gage association) — keep untouched
                    updated_lines.append(line)
                    replacing = True
                else:
                    if replace_count < len(new_rtk_list):
                        r, t, k = new_rtk_list[replace_count]
                        line = line[:46] + f"{r:<9.3f}{t:<9.3f}{k:<9.3f}" + line[73:]
                        replace_count += 1
                    # Even if more than 3 entries, keep rest unchanged
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    output_path = "C:/Users/chase/Downloads/TESTMODIFIED.inp"
    with open(output_path, "w") as f:
        f.writelines(updated_lines)

    print(f"RTK update complete. Modified file saved as '{output_path}'")


def evaluate(individual):
    rtk_values = [
        (individual[i], individual[i + 1], individual[i + 2]) for i in range(0, 9, 3)
    ]

    update_rtk_columns(rtk_values)

    # Step through the simulation and get data points from the outfall node with assigned Unit hydrograph
    inflows = np.array([])
    times = np.array([])
    # GOES BY SMALLEST AVIALBLE STEP OMEGA LOL
    with Simulation(
        "C:/Users/chase/Downloads/TESTMODIFIED.inp", sim_preconfig=sim_confg
    ) as sim:
        nodes = Nodes(sim)
        for step in sim:
            J1 = nodes["OUT"]
            inflows = np.append(inflows, J1.lateral_inflow)
            times = np.append(times, sim.current_time)

    # Compare peaks
    RTK_peak = max(inflows)
    Observed_peak = max(saved_iandi)
    peakerror = abs(Observed_peak - RTK_peak) / Observed_peak

    # compare volume / calculate volume
    # if i want to do this I have to convert DAtetime to a number...lotus should work.

    Observed_area = metrics.auc(saved_datelotus, saved_iandi)  # does this work?

    Lotustimes = ConvertToLotus(times)
    Simulated_area = metrics.auc(Lotustimes, inflows)  # Are these even the same?
    FlowScore = abs(Observed_area - Simulated_area) / Observed_area

    # compare shape.
    # to compare shape I think we are just going to use an RSE score.
    # 1st i need to assert that all of the time values are the same if they aren't interpolate the simulated to match observed
    # The Begining time should never be incorrect but I'd like for this program to handel irregular data well so we'll do it anyways.
    # assert times[0] == saved_date[0]
    # assert times[-1] == saved_date[-1]
    ShapeScore = RSE(saved_iandi, inflows)
    return (ShapeScore,)


def gen_rtk():
    val = [
        np.random.uniform(0.01, 0.5),
        np.random.uniform(0.5, 6),
        np.random.uniform(0.05, 1),
        np.random.uniform(0.01, 0.5),
        np.random.uniform(2, 12),
        np.random.uniform(0.05, 1),
        np.random.uniform(0.01, 0.5),
        np.random.uniform(5, 24),
        np.random.uniform(0.05, 1),
    ]
    return clampind(val)


def run_ga():

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=GENS,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return hof[0]


def clampind(individual):
    for i, (low, high) in enumerate(BOUNDS):
        individual[i] = max(min(individual[i], high), low)
    r_locid = [0, 3, 6]
    Rsum = sum(individual[i] for i in r_locid)
    if Rsum > 1:
        scale = 1 / Rsum
        for i in r_locid:
            individual[i] *= scale
    return individual


def custom_mutate(individual):
    tools.mutGaussian(individual, mu=0, sigma=0.1, indpb=0.2)
    return (clampind(individual),)


def displaybest(individual):
    # Step through and get values, plot, have fun
    rtk_values = [
        (individual[i], individual[i + 1], individual[i + 2]) for i in range(0, 9, 3)
    ]

    update_rtk_columns(rtk_values)

    # Step through the simulation and get data points from the outfall node with assigned Unit hydrograph
    inflows = np.array([])
    times = np.array([])
    # GOES BY SMALLEST AVIALBLE STEP OMEGA LOL
    with Simulation(
        "C:/Users/chase/Downloads/TESTMODIFIED.inp", sim_preconfig=sim_confg
    ) as sim:
        nodes = Nodes(sim)
        for step in sim:
            J1 = nodes["OUT"]
            inflows = np.append(inflows, J1.lateral_inflow)
            times = np.append(times, sim.current_time)
    plt.plot(inflows)
    plt.plot(saved_iandi)
    plt.xlabel("date(work in progress rn)")
    plt.ylabel("Flow (MGD)")
    plt.show()


df = pd.read_csv(
    "C:/Users/chase/Documents/GitHub/SSOAP-RTA-Optimization-Tool/ssoapExample/WWFwriteCOR.csv"
)
Ev = pd.read_csv(
    "C:/Users/chase/Documents/GitHub/SSOAP-RTA-Optimization-Tool/ssoapExample/EVENTS.txt",
    index_col=False,
    sep=" ",
)
toolbox = base.Toolbox()
# model info
node_id = ["OUT"]  # Replace with UI
hydrograph_name = "Analysis_2008-03-03"
base_inp = "C:/Users/chase/Downloads/TEST.inp"

# GA config
POP_SIZE = 10
GENS = 500
BOUNDS = [
    (0, 0.5),
    (0.5, 6),
    (0.05, 10),  # R1, T1, K1
    (0, 0.5),
    (0.5, 12),
    (0.05, 10),  # R2, T2, K2
    (0.0, 0.5),
    (0.5, 24),
    (0.05, 10),  # R3, T3, K3
]
# Create fitness and individual structure
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize error
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("individual", tools.initIterate, creator.Individual, gen_rtk)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Read the Flow monitor data "df" to find
ReportingStep = df["minute"][1] - df["minute"][0]
if ReportingStep < 0:
    ReportingStep = 60 + ReportingStep


# Get the RDII from SSOAP via written CSV file. we can worry about construction later.

saved_iandi = df["iandi"]
saved_datelotus = df["lotus date"]
saved_date = np.array([])
for index, row in df.iterrows():
    saved_date = np.append(saved_date, from_excel_ordinal(row["lotus date"]))

# we're doing event 1 for now because we are simply trying to set up the recurrsion loop.
StartDate = Ev["StartDate"].iloc(0)
print(StartDate[0])
StartTime = Ev["StartTime"].iloc(0)
EndDate = Ev["EndDate"].iloc(0)
EndTime = Ev["EndTime"].iloc(0)
Reportingdelta = timedelta(minutes=float(ReportingStep))  # 15 mintues
ReportingStr = str(Reportingdelta)
# Change swmm params to datetime data type
format_string = "%m/%d/%Y" + " %H:%M"
StartDateTime = datetime.strptime((StartDate[0] + " " + StartTime[0]), format_string)
EndDateTime = datetime.strptime((EndDate[0] + " " + EndTime[0]), format_string)
# adjust
adjStart = StartDateTime - Reportingdelta
adjEnd = EndDateTime + Reportingdelta
# Splice back out into day : time
adjStartDate = adjStart.strftime("%m-%d-%Y")
adjStartTime = adjStart.strftime("%H:%M:%S")
adjEndTime = adjEnd.strftime("%H:%M:%S")
adjEndDate = adjEnd.strftime("%m-%d-%Y")
# Because how how For step in Simulation is ran we have to set the start and end time 1 step before and one step after.
sim_confg = SimulationPreConfig()
sim_confg.input_file = "C:/Users/chase/Downloads/TEST.inp"
# Add start and end times.
sim_confg.add_update_by_token("OPTIONS", "START_DATE", 1, adjStartDate)
sim_confg.add_update_by_token("OPTIONS", "START_TIME", 1, adjStartTime)
sim_confg.add_update_by_token("OPTIONS", "REPORTING_START_DATE", 1, adjStartDate)
sim_confg.add_update_by_token("OPTIONS", "REPORTING_START_TIME", 1, adjStartTime)
sim_confg.add_update_by_token("OPTIONS", "END_DATE", 1, adjEndDate)
sim_confg.add_update_by_token("OPTIONS", "END_TIME", 1, adjEndTime)

# Change reporting times to reporting step
# SWMM uses the smallest avliable step in the config so we are going to change all of them
sim_confg.add_update_by_token("OPTIONS", "REPORT_STEP", 1, ReportingStr)
sim_confg.add_update_by_token("OPTIONS", "WET_STEP", 1, ReportingStr)
sim_confg.add_update_by_token("OPTIONS", "DRY_STEP", 1, ReportingStr)
sim_confg.add_update_by_token("OPTIONS", "ROUTING_STEP", 1, ReportingStr)
sim_confg.add_update_by_token("OPTIONS", "RULE_STEP", 1, ReportingStr)


best_rtk = run_ga()
displaybest(best_rtk)
print(
    "Best RTK triplets:",
    [(best_rtk[i], best_rtk[i + 1], best_rtk[i + 2]) for i in range(0, 9, 3)],
)
