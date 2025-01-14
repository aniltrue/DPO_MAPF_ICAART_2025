import datetime
import time
import numpy as np
import env
from agent.AgentFactory import AgentFactory
from agent.conflict_resolution import RandomProtocol
from agent.initial_path.LPStrategy import LPStrategy
from agent.repairing import EmptyRepairing, ACOUninformedRepairingStrategy
from env.Environment import Environment
import random
import pandas as pd

from math_model import LPFactory

MAP_SIZES = [10, 15, 20, 25]
AGENTS = [3, 6, 9, 12]
DENSITIES = [0.05, 0.1, 0.15, 0.15]
MAP_SEEDS = [1, 2, 3]
AGENT_SEEDS = [1]

INITIAL_ALGS = [LPStrategy]
REPAIR_ALGS = [EmptyRepairing]
PROTOCOLS = [RandomProtocol]

df = pd.DataFrame(
    columns=["MapSize", "NumberOfAgents", "Density", "MapSeed", "AgentSeed", "IsDynamic", "InitialAlg", "RepairingAlg",
             "Protocol", "Feasibility", "Result", "TimeStep", "ElapsedTime", "Cost", "OptimalValue",
             "OptimalElapsedTime", "NormalizedCost", "MaxDiffTokens", "AverageEMD"])

counter = 0

FILE_PATH = "lp_results.xlsx"

for map_size in MAP_SIZES:
    for number_of_agent in AGENTS:
        if map_size == 10 and number_of_agent == 12:
            continue

        for density in DENSITIES:
            for map_seed in MAP_SEEDS:
                map_seed_counter = 0
                org_map = env.map.StaticMap.random_map_factory(map_size, number_of_agent, density,
                                                               map_seed + map_seed_counter * 100)

                start_time = time.time()
                feasibility, cost = LPFactory.solve_for_objective(org_map)
                end_time = time.time()

                while not feasibility:
                    map_seed_counter += 1
                    org_map = env.map.StaticMap.random_map_factory(map_size, number_of_agent, density,
                                                                   map_seed + map_seed_counter * 100)

                    start_time = time.time()
                    feasibility, cost = LPFactory.solve_for_objective(org_map)
                    end_time = time.time()

                for agent_seed in AGENT_SEEDS:
                    row_exp = {"MapSize": map_size,
                               "NumberOfAgents": number_of_agent,
                               "Density": density,
                               "MapSeed": map_seed + map_seed_counter * 100,
                               "AgentSeed": agent_seed,
                               "IsDynamic": False,
                               "OptimalValue": cost,
                               "OptimalElapsedTime": end_time - start_time,
                               "MaxDiffTokens": 0,
                               "AverageEMD": 0}

                    for initial_alg in INITIAL_ALGS:
                        for repair_alg in REPAIR_ALGS:
                            for protocol_type in PROTOCOLS:
                                random.seed(agent_seed)
                                np.random.seed(agent_seed)

                                initial_algorithm = initial_alg()
                                repairing_algorithm = repair_alg()
                                protocol = protocol_type()

                                agents, elapsed_time = AgentFactory.generate_for_lp(org_map.clone(), initial_algorithm,
                                                                                    repairing_algorithm, protocol)

                                Env = Environment(org_map.clone(), agents)

                                result = Env.run()

                                row = row_exp.copy()
                                row["InitialAlg"] = initial_algorithm.name
                                row["RepairingAlg"] = repairing_algorithm.name
                                row["Protocol"] = protocol.name

                                row["Feasibility"] = result["feasibility"]
                                row["Result"] = result["result"]
                                row["TimeStep"] = result["time_step"]
                                row["ElapsedTime"] = result["elapsed_time"] + elapsed_time
                                row["Cost"] = result["cost"]
                                row["NormalizedCost"] = result["normalized_cost"]
                                row["MaxDiffTokens"] = result["max_token_diff"]
                                row["AverageEMD"] = result["average_emd"]

                                df.loc[counter] = row
                                counter += 1

                map_seed_counter = 0
                org_map = env.map.DPOMap.random_map_factory(map_size, number_of_agent, density,
                                                            map_seed + map_seed_counter * 100)

                start_time = time.time()
                feasibility, cost = LPFactory.solve_for_objective(org_map)
                end_time = time.time()

                while not feasibility:
                    map_seed_counter += 1
                    org_map = env.map.DPOMap.random_map_factory(map_size, number_of_agent, density,
                                                                map_seed + map_seed_counter * 100)

                    start_time = time.time()
                    feasibility, cost = LPFactory.solve_for_objective(org_map)
                    end_time = time.time()

                for agent_seed in AGENT_SEEDS:
                    row_exp = {"MapSize": map_size,
                               "NumberOfAgents": number_of_agent,
                               "Density": density,
                               "MapSeed": map_seed + map_seed_counter * 100,
                               "AgentSeed": agent_seed,
                               "IsDynamic": True,
                               "OptimalValue": cost,
                               "OptimalElapsedTime": end_time - start_time,
                               "MaxDiffTokens": 0,
                               "AverageEMD": 0}

                    for initial_alg in INITIAL_ALGS:
                        for repair_alg in REPAIR_ALGS:
                            for protocol_type in PROTOCOLS:
                                random.seed(agent_seed)
                                np.random.seed(agent_seed)

                                initial_algorithm = initial_alg()
                                repairing_algorithm = repair_alg()
                                protocol = protocol_type()

                                agents, elapsed_time = AgentFactory.generate_for_lp(org_map.clone(), initial_algorithm,
                                                                                    repairing_algorithm, protocol)

                                Env = Environment(org_map.clone(), agents)

                                result = Env.run()

                                row = row_exp.copy()
                                row["InitialAlg"] = initial_algorithm.name
                                row["RepairingAlg"] = repairing_algorithm.name
                                row["Protocol"] = protocol.name

                                row["Feasibility"] = result["feasibility"]
                                row["Result"] = result["result"]
                                row["TimeStep"] = result["time_step"]
                                row["ElapsedTime"] = result["elapsed_time"] + elapsed_time
                                row["Cost"] = result["cost"]
                                row["NormalizedCost"] = result["normalized_cost"]
                                row["MaxDiffTokens"] = result["max_token_diff"]
                                row["AverageEMD"] = result["average_emd"]

                                df.loc[counter] = row
                                counter += 1

                                print(result["result"])

                print(map_size, number_of_agent, density, map_seed, datetime.datetime.now())

                df.to_excel(FILE_PATH, sheet_name="EvaluationData")

df.to_excel(FILE_PATH, sheet_name="EvaluationData")
