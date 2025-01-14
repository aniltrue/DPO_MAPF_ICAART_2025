import datetime
import numpy as np
import env
from agent.AgentFactory import AgentFactory
from agent.conflict_resolution import RandomProtocol, FairTokenProtocol, ProbabilityBasedProtocol
from agent.initial_path import CBS
from agent.initial_path.aco import set_numba_seed
from agent.repairing import ACOInformedRepairingStrategy, EmptyRepairing
from env.Environment import Environment
import random
import pandas as pd

MAP_SIZES = [10, 15, 20, 25]
AGENTS = [3, 6, 9, 12]
DENSITIES = [0.05, 0.1, 0.15, 0.15]
MAP_SEEDS = [4, 5]
AGENT_SEEDS = [10]

INITIAL_ALGS = [CBS]
REPAIR_ALGS = [ACOInformedRepairingStrategy]
PROTOCOLS = [FairTokenProtocol]

INITIAL_B = [2, 5, 10]
END_B = [0.25, 0.5]
MAX_ITERATION = [150, 200, 250]
NUMBER_OF_ANTS = [25, 50, 75]
INFORMATION_MULTIPLIER = [1, 1.5, 2, 3, 5, 10]

df = pd.DataFrame(
    columns=["MapSize", "NumberOfAgents", "Density", "MapSeed", "AgentSeed", "IsDynamic", "InitialAlg", "RepairingAlg",
             "Protocol", "Feasibility", "Result", "TimeStep", "ElapsedTime", "Cost", "NormalizedCost", "MaxDiffTokens",
             "LowerBound", "AverageEMD", "InitialB", "EndB", "MaxIter", "NumberOfAnts", "InformationMultiplier"])

counter = 0

FILE_PATH = "hpo.xlsx"

for setting_counter in range(len(MAP_SIZES)):
    map_size = MAP_SIZES[setting_counter]
    number_of_agent = AGENTS[setting_counter]
    density = DENSITIES[setting_counter]

    for map_seed in MAP_SEEDS:
        map_seed_counter = 0

        # Check for feasibility of the map
        while True:
            org_map = env.map.StaticMap.random_map_factory(map_size, number_of_agent, density,
                                                           map_seed + map_seed_counter * 100)
            agents, elapsed_time = AgentFactory.generate(org_map, CBS(),
                                                         EmptyRepairing(), RandomProtocol(),
                                                         max_iteration=150,
                                                         initial_b=5.0,
                                                         end_b=0.25,
                                                         number_of_ants=50)

            Env = Environment(org_map.clone(), agents)

            result = Env.run()

            if result["feasibility"]:
                print("Started: ", map_size, number_of_agent, density, map_seed + map_seed_counter * 100,
                      datetime.datetime.now())
                break
            else:
                map_seed_counter += 1

        # Generate DPO Map
        org_map = env.map.DPOMap.random_map_factory(map_size, number_of_agent, density,
                                                    map_seed + map_seed_counter * 100)

        for agent_seed in AGENT_SEEDS:
            row_exp = {"MapSize": map_size,
                       "NumberOfAgents": number_of_agent,
                       "Density": density,
                       "AgentSeed": agent_seed,
                       "IsDynamic": True,
                       "MaxDiffTokens": 0,
                       "AverageEMD": 0}

            # Check for all combinations
            for initial_alg in INITIAL_ALGS:
                for repair_alg in REPAIR_ALGS:
                    for protocol_type in PROTOCOLS:
                        for number_of_ant in NUMBER_OF_ANTS:
                            for max_iteration in MAX_ITERATION:
                                for initial_b in INITIAL_B:
                                    for end_b in END_B:
                                        if initial_b <= end_b:
                                            continue

                                        for information_multiplier in INFORMATION_MULTIPLIER:
                                            random.seed(agent_seed)
                                            np.random.seed(agent_seed)
                                            set_numba_seed(agent_seed)

                                            initial_algorithm = initial_alg()
                                            repairing_algorithm = repair_alg(max_iter=max_iteration,
                                                                             initial_b=initial_b, end_b=end_b,
                                                                             number_of_ants=number_of_ant,
                                                                             information_multiplier=information_multiplier)
                                            protocol = protocol_type()

                                            agents, elapsed_time = AgentFactory.generate(org_map.clone(),
                                                                                         initial_algorithm,
                                                                                         repairing_algorithm, protocol,
                                                                                         max_iteration=max_iteration,
                                                                                         initial_b=initial_b,
                                                                                         end_b=end_b,
                                                                                         number_of_ants=number_of_ant)

                                            Env = Environment(org_map.clone(), agents)

                                            result = Env.run()

                                            row = row_exp.copy()
                                            row["InitialAlg"] = initial_algorithm.name
                                            row["RepairingAlg"] = repairing_algorithm.name
                                            row["Protocol"] = protocol.name

                                            row["MapSeed"] = map_seed + map_seed_counter * 100

                                            row["Feasibility"] = result["feasibility"]
                                            row["Result"] = result["result"]
                                            row["TimeStep"] = result["time_step"]
                                            row["ElapsedTime"] = result["elapsed_time"] + elapsed_time
                                            row["Cost"] = result["cost"]
                                            row["NormalizedCost"] = result["normalized_cost"]
                                            row["MaxDiffTokens"] = result["max_token_diff"]
                                            row["LowerBound"] = result["lower_bound"]
                                            row["AverageEMD"] = result["average_emd"]
                                            row["InitialB"] = initial_b
                                            row["EndB"] = end_b
                                            row["MaxIter"] = max_iteration
                                            row["NumberOfAnts"] = number_of_ant
                                            row["InformationMultiplier"] = information_multiplier

                                            df.loc[counter] = row
                                            counter += 1

        print(map_size, number_of_agent, density, map_seed, datetime.datetime.now())

        df.to_excel(FILE_PATH, sheet_name="HPOData")

df.to_excel(FILE_PATH, sheet_name="HPOData")
